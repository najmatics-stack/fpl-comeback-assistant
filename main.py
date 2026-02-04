#!/usr/bin/env python3
"""
FPL Comeback Assistant - Main Entry Point

Generates comprehensive FPL recommendations to help you climb the rankings.
"""

import argparse
import asyncio
import sys
from typing import List, Optional, Set

# Version and model info
VERSION = "1.0.0"
MODEL_NAME = "ians-model"
MODEL_DESCRIPTION = "Pure ownership (0.283 corr) + Mirror Ian Foster (#9, most consistent manager)"

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
from analysis.mirror_manager import (
    analyze_mirror, format_mirror_analysis,
    get_ian_weekly_moves, MIRROR_MANAGER_NAME,
    BudgetMirrorCandidate, find_budget_mirror_targets,
)
from output.recommendations import RecommendationEngine, TransferPlan, TransferRecommendation, CaptainPick


# ── ANSI Color Helpers ──────────────────────────────────────
# Works in Ghostty, iTerm2, and any modern terminal.
class C:
    """ANSI color codes for terminal output."""
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    # Foreground
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"
    # Bright foreground
    BRED    = "\033[91m"
    BGREEN  = "\033[92m"
    BYELLOW = "\033[93m"
    BBLUE   = "\033[94m"
    BMAGENTA= "\033[95m"
    BCYAN   = "\033[96m"


def c_header(text: str) -> str:
    """Bold cyan for section headers / phase titles."""
    return f"{C.BOLD}{C.BCYAN}{text}{C.RESET}"

def c_success(text: str) -> str:
    """Green for success messages."""
    return f"{C.BGREEN}{text}{C.RESET}"

def c_warn(text: str) -> str:
    """Yellow for warnings."""
    return f"{C.BYELLOW}{text}{C.RESET}"

def c_error(text: str) -> str:
    """Red for errors."""
    return f"{C.BRED}{text}{C.RESET}"

def c_debug(text: str) -> str:
    """Dim grey for debug/progress output."""
    return f"{C.DIM}{text}{C.RESET}"

def c_label(text: str) -> str:
    """Bold white for labels (player names, option keys)."""
    return f"{C.BOLD}{text}{C.RESET}"

def c_value(text: str) -> str:
    """Magenta for important values (prices, scores, ranks)."""
    return f"{C.BMAGENTA}{text}{C.RESET}"

def c_prompt(text: str) -> str:
    """Bold pink on dark background for user prompts — unmissable."""
    return f"\033[1m\033[38;5;205m\033[48;5;53m{text}\033[0m"

def c_option(text: str) -> str:
    """Blue for selectable options."""
    return f"{C.BBLUE}{text}{C.RESET}"


class UserCancelled(Exception):
    """Raised when the user types 'q' or 'quit' at any prompt."""
    pass


def checked_input(prompt_str: str = "") -> str:
    """Like input() but raises UserCancelled when the user types 'q' or 'quit'."""
    value = input(prompt_str)
    if value.strip().lower() in ("q", "quit"):
        raise UserCancelled()
    return value


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
        "--window",
        "-w",
        type=int,
        default=1,
        metavar="N",
        help="Number of GWs to evaluate (default 1, use 7 for robust sample)",
    )

    parser.add_argument(
        "--league-eval",
        type=int,
        metavar="GW",
        help="Evaluate your league's correlation vs model (specify end GW, uses --window)",
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

    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror Ian Foster (#9, most consistent manager) - shows transfers to match his squad",
    )

    parser.add_argument(
        "--mirror-threshold",
        type=int,
        default=4,
        help="Number of transfers above which to recommend Free Hit (default: 4)",
    )

    parser.add_argument(
        "--lineup",
        action="store_true",
        help="Set your starting 11 and bench order (requires login)",
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

    print("\n" + c_header("=" * 60))
    print(c_header("  PHASE 0: SETTINGS"))
    print(c_header("=" * 60))
    print(
        f"\n  max_hits={settings['max_hits']} | "
        f"min_gain_free={settings['min_gain_free']} | "
        f"min_gain_hit={settings['min_gain_hit']} | "
        f"max_transfers={settings['max_transfers']} | "
        f"risk={settings['risk']}"
    )

    choice = checked_input(c_prompt("\nAdjust? [Enter=skip, s=settings, q=quit]: ")).strip().lower()
    if choice != "s":
        return settings

    setting_keys = list(settings.keys())
    while True:
        print("\nSettings:")
        for i, key in enumerate(setting_keys, 1):
            print(f"  {i}. {key} = {settings[key]}")
        print()
        cmd = checked_input("Change (<number> <value>, or Enter=done, q=quit): ").strip()
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
    print("\n" + c_header("=" * 60))
    print(c_header("  PHASE 0.5: CHIP SELECTION"))
    print(c_header("=" * 60))

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
    print(c_prompt(f"\n  Play a chip? [Enter=none, {prompt_str}, q=quit]: "), end="")
    choice = checked_input().strip().lower()

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
    print("\n" + c_header("=" * 60))
    print(c_header("  PHASE 0.75: LOCK PLAYERS"))
    print(c_header("=" * 60))

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
    choice = checked_input(c_prompt("  [Enter=none, e.g. 1,3,5, q=quit]: ")).strip()

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


def _build_mirror_transfers(
    recommender: RecommendationEngine,
    fpl: FPLDataFetcher,
    squad_ids: List[int],
    target_squad: List[int],
    reason: str = "Mirror Ian Foster",
) -> List[TransferRecommendation]:
    """Build transfer list to transform squad_ids into target_squad.

    Matches outgoing/incoming players by position to satisfy FPL's
    same-position transfer requirement.
    """
    current_set = set(squad_ids)
    target_set = set(target_squad)

    # Group outgoing and incoming by position
    outs_by_pos: dict = {}
    ins_by_pos: dict = {}

    for pid in squad_ids:
        if pid not in target_set:
            p = fpl.get_player(pid)
            if p:
                sp = recommender.scorer.score_player(p)
                outs_by_pos.setdefault(p.position, []).append(sp)

    for pid in target_squad:
        if pid not in current_set:
            p = fpl.get_player(pid)
            if p:
                sp = recommender.scorer.score_player(p)
                ins_by_pos.setdefault(p.position, []).append(sp)

    transfers = []
    for pos in ["GKP", "DEF", "MID", "FWD"]:
        pos_outs = outs_by_pos.get(pos, [])
        pos_ins = ins_by_pos.get(pos, [])
        # Sort: outs by score ascending, ins by score descending
        # so best incoming replaces worst outgoing
        pos_outs.sort(key=lambda sp: sp.overall_score)
        pos_ins.sort(key=lambda sp: sp.overall_score, reverse=True)
        for out_sp, in_sp in zip(pos_outs, pos_ins):
            transfers.append(TransferRecommendation(
                player_out=out_sp,
                player_in=in_sp,
                score_gain=in_sp.overall_score - out_sp.overall_score,
                price_diff=in_sp.player.price - out_sp.player.price,
                reason=reason,
            ))

    return transfers


def _budget_constrained_mirror(
    ian_squad: List[int],
    squad_ids: List[int],
    total_budget: float,
    scorer: "PlayerScorer",
    fpl: FPLDataFetcher,
    ian_captain: Optional[int] = None,
) -> Optional[List[int]]:
    """Downgrade a target squad to fit within budget.

    Strategy — protect the starting 11, sacrifice the bench:
      1. Bench players (indices 11-14) are downgraded first, auto-picking the
         cheapest valid option without prompting.
      2. Starting 11 (indices 0-10) are only touched if the bench downgrades
         weren't enough. The user picks from 3 options for each starter.
      3. Captain is never downgraded.

    Returns modified squad list or None if budget can't be met.
    """
    current_set = set(squad_ids)
    adjusted_squad = list(ian_squad)

    # The squad list from the picks endpoint preserves order:
    # indices 0-10 = starting XI, 11-14 = bench
    bench_indices = set(range(11, 15))

    def squad_cost(squad: List[int]) -> float:
        return sum((fpl.get_player(pid).price if fpl.get_player(pid) else 0.0) for pid in squad)

    cost = squad_cost(adjusted_squad)
    if cost <= total_budget + 0.05:
        return adjusted_squad  # Already fits

    # Track team counts for 3-per-team rule
    team_counts: dict = {}
    for pid in adjusted_squad:
        p = fpl.get_player(pid)
        if p:
            team_counts[p.team_id] = team_counts.get(p.team_id, 0) + 1

    swaps_made = []

    def _find_valid_cheaper(player, adjusted_squad, aggressive=False):
        """Find cheaper same-position alternatives.
        aggressive=True: sort by price ascending (cheapest first) for bench.
        aggressive=False: sort by score descending (best first) for starters.
        """
        candidates = scorer.get_top_players(
            position=player.position,
            max_price=player.price - 0.1,
            limit=50 if aggressive else 30,
            exclude_unavailable=True,
        )
        adjusted_set = set(adjusted_squad)
        valid = []
        for candidate in candidates:
            cp = candidate.player
            if cp.id in adjusted_set:
                continue
            tc = team_counts.get(cp.team_id, 0)
            if cp.team_id != player.team_id and tc >= 3:
                continue
            valid.append(candidate)

        if aggressive:
            # Sort by price ascending — maximize savings for bench
            valid.sort(key=lambda c: c.player.price)
        # else: already sorted by score from get_top_players

        return valid[:5] if aggressive else valid[:3]

    def _do_swap(pid, player, chosen_player):
        nonlocal cost
        idx = adjusted_squad.index(pid)
        adjusted_squad[idx] = chosen_player.id

        team_counts[player.team_id] = team_counts.get(player.team_id, 1) - 1
        if chosen_player.team_id != player.team_id:
            team_counts[chosen_player.team_id] = team_counts.get(chosen_player.team_id, 0) + 1

        saving = player.price - chosen_player.price
        cost -= saving
        swaps_made.append((player.web_name, chosen_player.web_name, player.price, chosen_player.price, saving))

    # ── PASS 1: Bench downgrades (automatic, cheapest option) ──
    bench_targets = []
    for i in bench_indices:
        if i >= len(ian_squad):
            continue
        pid = ian_squad[i]
        p = fpl.get_player(pid)
        if p and pid != ian_captain:
            sp = scorer.score_player(p)
            bench_targets.append((pid, p, sp, i))

    # Sort bench by price descending (most expensive bench player first = most savings)
    bench_targets.sort(key=lambda x: x[1].price, reverse=True)

    for pid, player, scored, squad_idx in bench_targets:
        if cost <= total_budget + 0.05:
            break

        valid = _find_valid_cheaper(player, adjusted_squad, aggressive=True)
        if not valid:
            continue

        # Auto-pick cheapest valid option for bench
        chosen = valid[0]
        cp = chosen.player
        saving = player.price - cp.price
        print(c_debug(f"  Bench: {player.web_name} (£{player.price}m) → {cp.web_name} (£{cp.price}m)  [saves £{saving:.1f}m]"))
        _do_swap(pid, player, cp)

    if cost <= total_budget + 0.05:
        if swaps_made:
            print(c_success(f"\n  ✓ Bench downgrades only ({len(swaps_made)} swaps) — starting 11 untouched"))
            print(c_debug(f"  Adjusted cost: £{cost:.1f}m (budget: £{total_budget:.1f}m)"))
        return adjusted_squad

    # ── PASS 2: Starter downgrades (interactive, user picks) ──
    print(c_warn(f"\n  Bench downgrades saved £{sum(s[4] for s in swaps_made):.1f}m but still £{cost - total_budget:.1f}m over."))
    print(c_warn("  Need to downgrade starter(s):"))

    starter_targets = []
    for i in range(11):
        if i >= len(ian_squad):
            continue
        pid = adjusted_squad[i]  # Use adjusted (bench already swapped)
        if pid in current_set:
            continue  # Already own this player, can't downgrade
        if pid == ian_captain:
            continue
        p = fpl.get_player(pid)
        if p:
            sp = scorer.score_player(p)
            starter_targets.append((pid, p, sp))

    # Sort by score ascending (lowest-impact starter first)
    starter_targets.sort(key=lambda x: x[2].overall_score)

    for pid, player, scored in starter_targets:
        if cost <= total_budget + 0.05:
            break

        valid = _find_valid_cheaper(player, adjusted_squad, aggressive=False)
        if not valid:
            continue

        shortfall = cost - total_budget
        print(f"\n  Downgrade needed: {c_label(player.web_name)} ({player.position}, £{player.price}m, score {scored.overall_score:.1f})")
        print(f"  Still {c_warn(f'£{shortfall:.1f}m')} over budget. Pick a replacement:")
        for i, cand in enumerate(valid, 1):
            cp = cand.player
            saving = player.price - cp.price
            print(f"    {i}. {cp.web_name:15} ({cp.team}) £{cp.price}m | Score: {cand.overall_score:.1f} | saves £{saving:.1f}m")

        pick = checked_input(c_prompt(f"  [1-{len(valid)}, Enter=1, q=quit]: ")).strip()
        try:
            pick_idx = int(pick) - 1 if pick else 0
            if pick_idx < 0 or pick_idx >= len(valid):
                pick_idx = 0
        except ValueError:
            pick_idx = 0

        chosen = valid[pick_idx]
        _do_swap(pid, player, chosen.player)

    if cost > total_budget + 0.05:
        print(c_error(f"  ⚠️  Could not fit squad within budget (£{cost:.1f}m > £{total_budget:.1f}m)"))
        return None

    # Final summary
    bench_swaps = [s for s in swaps_made if s[0] in [bt[1].web_name for bt in bench_targets]]
    starter_swaps = [s for s in swaps_made if s not in bench_swaps]
    print(f"\n  Budget adjustments ({len(swaps_made)} downgrades):")
    if bench_swaps:
        print(c_debug(f"  Bench ({len(bench_swaps)}):"))
        for orig, repl, orig_price, repl_price, saving in bench_swaps:
            print(c_debug(f"    {orig} (£{orig_price}m) → {repl} (£{repl_price}m)  [saves £{saving:.1f}m]"))
    if starter_swaps:
        print(c_warn(f"  Starters ({len(starter_swaps)}):"))
        for orig, repl, orig_price, repl_price, saving in starter_swaps:
            print(f"    {orig} (£{orig_price}m) → {repl} (£{repl_price}m)  [saves £{saving:.1f}m]")
    print(f"  Adjusted squad cost: £{cost:.1f}m (budget: £{total_budget:.1f}m)")

    return adjusted_squad


def prompt_free_hit_squad(
    recommender: RecommendationEngine,
    fpl: FPLDataFetcher,
    squad_ids: List[int],
    ian_squad: Optional[List[int]] = None,
    ian_captain: Optional[int] = None,
    budget_info: Optional[dict] = None,
    budget_candidates: Optional[List["BudgetMirrorCandidate"]] = None,
) -> tuple:
    """Phase 1-FH: Build and display optimal Free Hit squad.

    Args:
        budget_info: Dict from get_squad_with_budget() with selling prices + bank.
                     If None, falls back to market prices (less accurate).

    Returns (chosen_transfers, new_squad_ids) or ([], squad_ids) if skipped.
    """
    print("\n" + c_header("=" * 60))
    print(c_header("  PHASE 1-FH: FREE HIT SQUAD"))
    print(c_header("=" * 60))

    # Calculate total budget — prefer selling prices (accurate) over market prices
    if budget_info:
        total_budget = budget_info["total_budget"]
        print(f"\n  Budget: £{total_budget:.1f}m (selling prices + £{budget_info['bank']:.1f}m bank)")
    else:
        total_budget = 0.0
        for pid in squad_ids:
            player = fpl.get_player(pid)
            if player:
                total_budget += player.price
        if total_budget < 50.0:
            total_budget = 100.0  # Fallback
        print(f"\n  Budget: £{total_budget:.1f}m (market prices — selling prices may be lower)")
        print("  ⚠️  Could not fetch selling prices; actual budget may differ")

    # Helper to print a squad grouped by position
    def _print_squad(squad: List[int], captain_id: Optional[int] = None, indent: str = "    "):
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_players = []
            for pid in squad:
                p = fpl.get_player(pid)
                if p and p.position == pos:
                    cap_mark = " (C)" if pid == captain_id else ""
                    pos_players.append(f"{p.web_name}{cap_mark}")
            if pos_players:
                print(f"{indent}{pos}: {', '.join(pos_players)}")

    # Calculate costs
    ian_cost = 0.0
    ian_over_budget = False
    if ian_squad and len(ian_squad) == 15:
        for pid in ian_squad:
            p = fpl.get_player(pid)
            if p:
                ian_cost += p.price
        ian_over_budget = ian_cost > total_budget + 0.05

    current_set = set(squad_ids)
    has_candidates = budget_candidates and len(budget_candidates) > 0
    has_ian = ian_squad and len(ian_squad) == 15

    # --- Split candidates into fits-budget vs needs-downgrades ---
    fits_budget = []
    needs_downgrade = []
    if has_candidates:
        for cand in budget_candidates:
            if cand.squad_cost <= total_budget + 0.05:
                fits_budget.append(cand)
            else:
                needs_downgrade.append(cand)
        fits_budget = fits_budget[:3]
        needs_downgrade = needs_downgrade[:3]

    # Number candidates sequentially across both groups
    all_numbered = fits_budget + needs_downgrade  # [1..N]

    def _print_candidate(idx: int, cand, show_cost_tag: bool = False):
        cap_name = "?"
        cap_p = fpl.get_player(cand.captain)
        if cap_p:
            cap_name = cap_p.web_name
        over = cand.squad_cost - total_budget
        if over > 0.05:
            cost_str = c_warn(f"£{cand.squad_cost:.1f}m") + c_warn(f" (+£{over:.1f}m)")
        else:
            cost_str = c_success(f"£{cand.squad_cost:.1f}m")
        print(c_option(f"  [{idx}] ") + c_label(f"#{cand.rank} \"{cand.team_name}\"") + f" by {cand.manager_name}")
        print(f"      Points: {c_value(f'{cand.total_points:,}')} | GW: {c_value(str(cand.gw_points))} | Cost: {cost_str} | Captain: {c_label(cap_name)}")
        print(f"      Overlap: {cand.overlap_count}/15 ({cand.transfers_needed} transfers)")
        _print_squad(cand.squad, cand.captain, indent="      ")
        print()

    num_idx = 1
    if fits_budget:
        print(c_header(f"\n  --- FITS YOUR BUDGET (No Downgrades) ---"))
        print(c_debug(f"  Budget: £{total_budget:.1f}m\n"))
        for cand in fits_budget:
            _print_candidate(num_idx, cand)
            num_idx += 1

    if needs_downgrade:
        print(c_header(f"\n  --- NEEDS MINOR DOWNGRADES ---"))
        print(c_debug(f"  Slightly over budget — will auto-pick cheaper alternatives\n"))
        for cand in needs_downgrade:
            _print_candidate(num_idx, cand)
            num_idx += 1

    # --- Show Ian Foster ---
    if has_ian:
        ian_set = set(ian_squad)
        ian_transfers = len(ian_set - current_set)
        if ian_over_budget:
            shortfall = ian_cost - total_budget
            print(c_header(f"  --- IAN FOSTER (Needs Downgrades) ---"))
            print(f"  Cost: {c_warn(f'£{ian_cost:.1f}m')} ({c_warn(f'£{shortfall:.1f}m over budget')}) | Transfers: {ian_transfers} (will auto-downgrade)")
        else:
            print(c_header(f"\n  --- IAN FOSTER (Recommended) ---"))
            print(f"  Most consistent manager, never below #1,248")
            print(f"  Cost: {c_success(f'£{ian_cost:.1f}m')} (within budget) | Transfers: {ian_transfers}")
        _print_squad(ian_squad, ian_captain, indent="    ")
        print()

    # Build prompt options
    num_cands = len(all_numbered)
    parts = []
    if num_cands:
        cand_keys = "/".join(str(i) for i in range(1, num_cands + 1))
        parts.append(f"{cand_keys}=pick manager")
    if has_ian:
        ian_label = "i=Ian (downgrade)" if ian_over_budget else "i=Ian"
        parts.append(ian_label)
    parts.extend(["m=model", "s=skip", "q=quit"])

    print(c_prompt(f"\n  [{', '.join(parts)}]: "), end="")
    choice = checked_input().strip().lower()

    # Handle scanned candidate selection
    if num_cands and choice.isdigit() and 1 <= int(choice) <= num_cands:
        cand = all_numbered[int(choice) - 1]
        target_squad = list(cand.squad)
        reason = f"Mirror #{cand.rank} {cand.manager_name}"

        # If candidate is over real budget, auto-downgrade
        if cand.squad_cost > total_budget + 0.05:
            print(c_warn(f"\n  Squad is £{cand.squad_cost - total_budget:.1f}m over budget — picking downgrades..."))
            adjusted = _budget_constrained_mirror(
                target_squad, squad_ids, total_budget,
                recommender.scorer, fpl, cand.captain,
            )
            if adjusted is None:
                print(c_error("  ❌ Cannot fit squad within budget even with downgrades"))
                print(c_warn("     Falling back to model's squad..."))
                choice = "m"
            else:
                target_squad = adjusted

        if choice != "m":
            transfers = _build_mirror_transfers(
                recommender, fpl, squad_ids, target_squad, reason=reason,
            )
            print(c_success(f"\n  ✓ Using #{cand.rank} {cand.manager_name}'s squad ({len(transfers)} transfers)"))
            return transfers, target_squad

    # Handle Ian selection
    if choice == "i" and has_ian:
        if ian_over_budget:
            adjusted = _budget_constrained_mirror(
                ian_squad, squad_ids, total_budget,
                recommender.scorer, fpl, ian_captain,
            )
            if adjusted is None:
                print(c_error("  ❌ Cannot fit Ian's squad within budget even with downgrades"))
                print(c_warn("     Falling back to model's squad..."))
                choice = "m"
            else:
                transfers = _build_mirror_transfers(
                    recommender, fpl, squad_ids, adjusted,
                )
                print(c_success(f"\n  ✓ Using Ian Foster's squad with downgrades ({len(transfers)} transfers)"))
                return transfers, adjusted
        else:
            transfers = _build_mirror_transfers(
                recommender, fpl, squad_ids, ian_squad,
            )
            print(c_success(f"\n  ✓ Using Ian Foster's squad ({len(transfers)} transfers)"))
            return transfers, ian_squad

    if choice == "s":
        print(c_debug("  Skipping Free Hit."))
        return [], squad_ids

    # Fall through to model's squad
    print()

    # Model's optimal squad (Option B/C or fallback)
    print("  --- MODEL'S OPTIMAL SQUAD ---")

    squad, transfers = recommender.get_free_hit_squad(squad_ids, total_budget)

    if not squad or len(squad) < 15:
        print("  Could not build a full 15-player squad.")
        return [], squad_ids

    new_squad_ids = [sp.player.id for sp in squad]
    print(recommender.format_free_hit_squad(squad, transfers, total_budget))

    while True:
        print(f"\n  [Enter=accept, e=edit, s=skip, q=quit]: ", end="")
        choice = checked_input().strip().lower()

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

                cmd = checked_input("\nedit-fh> ").strip()
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
    print("\n" + c_header("=" * 60))
    print(c_header("  PHASE 1: TRANSFER PLAN"))
    print(c_header("=" * 60))

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
        f"a/b/c=select, e=edit, s=skip, q=quit]: ",
        end="",
    )
    choice = checked_input().strip().lower()

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

        cmd = checked_input("\nedit> ").strip()
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
    print("\n" + c_header("=" * 60))
    print(c_header("  PHASE 2: CAPTAIN"))
    print(c_header("=" * 60))

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
        f"\nCaptain [Enter=#1, 1-{len(captain_picks)}=select, o=override, q=quit]: ",
        end="",
    )
    choice = checked_input().strip().lower()

    if choice == "o":
        name_query = checked_input("  Captain name: ").strip()
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
        vc_choice = checked_input("  > ").strip()
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


# Valid FPL formations: (DEF, MID, FWD)
VALID_FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2), (4, 5, 1),
    (5, 3, 2), (5, 4, 1), (5, 2, 3),
]


def _best_lineup(
    squad_ids: List[int],
    fpl: FPLDataFetcher,
    scorer: "PlayerScorer",
    captain_id: Optional[int] = None,
) -> tuple:
    """Pick the optimal starting 11 from a 15-man squad.

    Returns (starting_11_ids, bench_4_ids) where bench is ordered by score
    descending (best sub first).
    """
    # Score all players
    players = []
    for pid in squad_ids:
        p = fpl.get_player(pid)
        if p:
            sp = scorer.score_player(p)
            players.append((pid, p, sp))

    by_pos = {}
    for pid, p, sp in players:
        by_pos.setdefault(p.position, []).append((pid, p, sp))

    # Sort each position by score descending
    for pos in by_pos:
        by_pos[pos].sort(key=lambda x: x[2].overall_score, reverse=True)

    gkps = by_pos.get("GKP", [])
    defs = by_pos.get("DEF", [])
    mids = by_pos.get("MID", [])
    fwds = by_pos.get("FWD", [])

    best_score = -1
    best_starting = None
    best_bench = None

    for n_def, n_mid, n_fwd in VALID_FORMATIONS:
        if n_def > len(defs) or n_mid > len(mids) or n_fwd > len(fwds) or len(gkps) < 1:
            continue

        starting = [gkps[0]] + defs[:n_def] + mids[:n_mid] + fwds[:n_fwd]
        total = sum(sp.overall_score for _, _, sp in starting)

        if total > best_score:
            best_score = total
            best_starting = starting

    if not best_starting:
        # Fallback: just pick top 11 by score
        all_sorted = sorted(players, key=lambda x: x[2].overall_score, reverse=True)
        best_starting = all_sorted[:11]

    starting_ids = set(pid for pid, _, _ in best_starting)
    bench = [(pid, p, sp) for pid, p, sp in players if pid not in starting_ids]
    # Bench order: GKP first (position 12 required by FPL), then outfield by score desc
    bench.sort(key=lambda x: (x[1].position != "GKP", -x[2].overall_score))

    return (
        [pid for pid, _, _ in best_starting],
        [pid for pid, _, _ in bench],
    )


def prompt_lineup(
    fpl: FPLDataFetcher,
    scorer: "PlayerScorer",
    squad_ids: List[int],
    captain_id: Optional[int] = None,
    vc_id: Optional[int] = None,
) -> Optional[tuple]:
    """Phase 3: Set starting 11 and bench order.

    Returns (starting_11, bench_4, captain_id, vc_id) or None if skipped.
    """
    print(c_header("\n" + "=" * 60))
    print(c_header("  PHASE 3: STARTING LINEUP"))
    print(c_header("=" * 60))

    # Auto-optimize lineup
    starting, bench = _best_lineup(squad_ids, fpl, scorer, captain_id)

    # If no captain set, pick the highest-scored starter (excluding GKP)
    if not captain_id:
        best_score = -1
        for pid in starting:
            p = fpl.get_player(pid)
            if p and p.position != "GKP":
                sp = scorer.score_player(p)
                if sp.overall_score > best_score:
                    best_score = sp.overall_score
                    captain_id = pid
    if not vc_id:
        # Second-best outfield starter
        best_score = -1
        for pid in starting:
            p = fpl.get_player(pid)
            if p and p.position != "GKP" and pid != captain_id:
                sp = scorer.score_player(p)
                if sp.overall_score > best_score:
                    best_score = sp.overall_score
                    vc_id = pid
    if not vc_id:
        vc_id = captain_id

    def _display_lineup(starting, bench):
        print(f"\n  {c_label('STARTING XI:')}")
        idx = 1
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_players = []
            for pid in starting:
                p = fpl.get_player(pid)
                if p and p.position == pos:
                    sp = scorer.score_player(p)
                    cap = " (C)" if pid == captain_id else (" (V)" if pid == vc_id else "")
                    pos_players.append((idx, p, sp, cap))
                    idx += 1
            if pos_players:
                print(f"  {c_debug(pos)}:")
                for i, p, sp, cap in pos_players:
                    print(f"   {i:2}. {p.web_name:15} ({p.team:3}) £{p.price}m | Score: {c_value(f'{sp.overall_score:.1f}')}{cap}")

        # Formation
        pos_counts = {}
        for pid in starting:
            p = fpl.get_player(pid)
            if p:
                pos_counts[p.position] = pos_counts.get(p.position, 0) + 1
        formation = f"{pos_counts.get('DEF', 0)}-{pos_counts.get('MID', 0)}-{pos_counts.get('FWD', 0)}"
        print(f"\n  Formation: {c_value(formation)}")

        print(f"\n  {c_label('BENCH')} (sub order):")
        for i, pid in enumerate(bench, 12):
            p = fpl.get_player(pid)
            if p:
                sp = scorer.score_player(p)
                print(f"   {i}. {p.web_name:15} ({p.team:3}) £{p.price}m | Score: {c_debug(f'{sp.overall_score:.1f}')}")

    _display_lineup(starting, bench)

    while True:
        print(c_prompt(f"\n  [Enter=accept, swap <a> <b>, s=skip, q=quit]: "), end="")
        choice = checked_input().strip().lower()

        if choice == "s":
            print(c_debug("  Skipping lineup."))
            return None

        if not choice:
            print(c_success(f"  ✓ Lineup accepted"))
            return starting, bench, captain_id, vc_id

        if choice.startswith("swap"):
            parts = choice.split()
            if len(parts) != 3:
                print("  Usage: swap <number> <number>  (e.g. swap 3 12)")
                continue
            try:
                a_idx = int(parts[1])
                b_idx = int(parts[2])
                all_players = starting + bench
                if a_idx < 1 or a_idx > 15 or b_idx < 1 or b_idx > 15:
                    print(f"  Numbers must be 1-15")
                    continue

                a_pid = all_players[a_idx - 1]
                b_pid = all_players[b_idx - 1]

                # Swap them
                all_players[a_idx - 1] = b_pid
                all_players[b_idx - 1] = a_pid

                new_starting = all_players[:11]
                new_bench = all_players[11:]

                # Validate formation
                pos_counts = {}
                for pid in new_starting:
                    p = fpl.get_player(pid)
                    if p:
                        pos_counts[p.position] = pos_counts.get(p.position, 0) + 1

                n_gkp = pos_counts.get("GKP", 0)
                n_def = pos_counts.get("DEF", 0)
                n_mid = pos_counts.get("MID", 0)
                n_fwd = pos_counts.get("FWD", 0)

                if n_gkp != 1:
                    print(c_warn(f"  Invalid: need exactly 1 GKP in starting XI, got {n_gkp}"))
                    continue
                if (n_def, n_mid, n_fwd) not in VALID_FORMATIONS:
                    print(c_warn(f"  Invalid formation: {n_def}-{n_mid}-{n_fwd}"))
                    continue

                starting = new_starting
                # Enforce bench GKP at position 12 (FPL requirement)
                new_bench.sort(key=lambda pid: (fpl.get_player(pid).position != "GKP" if fpl.get_player(pid) else True))
                bench = new_bench
                a_name = fpl.get_player(a_pid).web_name if fpl.get_player(a_pid) else "?"
                b_name = fpl.get_player(b_pid).web_name if fpl.get_player(b_pid) else "?"
                print(c_success(f"  Swapped: {a_name} ↔ {b_name}"))
                _display_lineup(starting, bench)

            except (ValueError, IndexError):
                print("  Usage: swap <number> <number>")
        else:
            print("  Commands: swap <a> <b>, Enter=accept, s=skip, q=quit")

    return None


async def prompt_review_and_execute(
    fpl_actions: "FPLActions",
    transfers: List[TransferRecommendation],
    captain_id: Optional[int],
    vc_id: Optional[int],
    chip_to_play,
    current_gw: int,
    lineup: Optional[tuple] = None,
):
    """Phase 4: Final review and execute.

    Uses the already-authenticated FPLActions session from the initial login.
    ``lineup`` is (starting_11, bench_4, captain_id, vc_id) or None.
    """
    print("\n" + c_header("=" * 60))
    print(c_header("  PHASE 4: REVIEW & EXECUTE"))
    print(c_header("=" * 60))

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

    if lineup:
        actions.append("Set starting lineup & bench order")

    if not actions:
        print("\n  No actions to execute.")
        return

    print("\n  Planned actions:")
    for i, action in enumerate(actions, 1):
        print(f"   {i}. {action}")

    print(c_prompt("\n  Execute? [Enter/y=yes, n=cancel, q=quit]: "), end="")
    confirm = checked_input().strip().lower()
    if confirm == "n":
        print("  Cancelled.")
        return

    # Execute using the existing authenticated session
    all_ok = True

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

    # Set lineup (starting 11 + bench order)
    if lineup:
        lu_starting, lu_bench, lu_cap, lu_vc = lineup
        if not await fpl_actions.set_lineup(lu_starting, lu_bench, lu_cap, lu_vc):
            all_ok = False

    if all_ok:
        print(c_success("\n✓ All actions executed successfully!"))
    else:
        print(c_error("\n⚠️  Some actions failed — check output above"))


def display_model_performance():
    """Display model vs ownership baseline from latest evaluation."""
    from analysis.evaluator import WEIGHTS_FILE
    import json

    if not WEIGHTS_FILE.exists():
        return

    try:
        with open(WEIGHTS_FILE) as f:
            history = json.load(f)

        if not history:
            return

        # Find the best entry: prefer multi-GW (higher window), then latest GW
        def sort_key(key: str):
            window = history[key].get("window", 1)
            if "-" in key:
                gw = int(key.split("-")[0])
            else:
                gw = int(key)
            return (window, gw)  # Higher window first, then higher GW

        latest_key = max(history.keys(), key=sort_key)
        entry = history[latest_key]

        model_corr = entry.get("overall_correlation", 0)
        ownership_corr = entry.get("correlations", {}).get("ownership", 0)
        # Try new field first, fall back to correlations.ownership
        ownership_baseline = entry.get("ownership_baseline", ownership_corr)
        window = entry.get("window", 1)

        if model_corr and ownership_baseline:
            advantage = model_corr - ownership_baseline
            pct = (advantage / ownership_baseline * 100) if ownership_baseline > 0 else 0

            window_label = f"{window}-week" if window > 1 else "single-GW"
            print(f"\n📊 Model Performance ({window_label} evaluation):")
            print(f"   Our Model:      {model_corr:.3f} correlation")
            print(f"   Ownership Only: {ownership_baseline:.3f} correlation")
            if advantage >= 0:
                print(f"   Advantage:      +{advantage:.3f} ({pct:+.1f}% better)")
            else:
                print(f"   Advantage:      {advantage:.3f} ({pct:.1f}% worse)")

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass


async def interactive_auto_mode(
    recommender: RecommendationEngine,
    fpl: FPLDataFetcher,
    squad_ids: List[int],
    available_chips: List[str],
    team_id: int,
    current_gw: int,
):
    """Interactive auto-pilot mode (Phases 0–4)."""
    print("\n" + c_header("=" * 60))
    print(c_header("  AUTO-PILOT MODE (Interactive)"))
    print(c_header("=" * 60))

    # Display model performance summary
    display_model_performance()

    # Show mirror references and save squads for Free Hit option
    ian_squad = None
    ian_captain = None

    player_names = {p.id: p.web_name for p in fpl.get_all_players()}

    # --- Ian Foster ---
    print("\n" + c_header("=" * 60))
    print(c_header("  MIRROR REFERENCE: Ian Foster (Most Consistent)"))
    print(c_header("=" * 60))
    try:
        mirror_analysis = await analyze_mirror(
            your_squad=squad_ids,
            current_gw=current_gw,
            free_hit_threshold=4,
        )
        if mirror_analysis:
            ian_squad = mirror_analysis.target_squad
            ian_captain = mirror_analysis.target_captain

            captain_name = player_names.get(mirror_analysis.target_captain, "Unknown")

            print(f"\n  Ian's Rank: #{mirror_analysis.target_rank:,} | Points: {mirror_analysis.target_points}")
            print(f"  Transfers to match: {mirror_analysis.transfers_needed}")

            if mirror_analysis.transfers_needed > 0:
                print(f"\n  Quick diff:")
                for pid in mirror_analysis.players_to_sell[:3]:
                    name = player_names.get(pid, f"ID:{pid}")
                    print(f"    OUT: {name}")
                if mirror_analysis.transfers_needed > 3:
                    print(f"    ... and {mirror_analysis.transfers_needed - 3} more")
                for pid in mirror_analysis.players_to_buy[:3]:
                    name = player_names.get(pid, f"ID:{pid}")
                    print(f"    IN:  {name}")
                if mirror_analysis.transfers_needed > 3:
                    print(f"    ... and {mirror_analysis.transfers_needed - 3} more")

            if mirror_analysis.recommend_free_hit:
                print(f"\n  ⚠️  Consider FREE HIT to mirror Ian ({mirror_analysis.transfers_needed} transfers)")

            print(f"\n  Ian's captain: {captain_name}")

            moves = await get_ian_weekly_moves(current_gw)
            if moves and moves['transfers_in']:
                print(f"  Ian's transfer: {player_names.get(moves['transfers_out'][0], '?')} → {player_names.get(moves['transfers_in'][0], '?')}")

            print(f"\n  (Run --mirror for full analysis)")
    except Exception as e:
        print(f"  Could not fetch Ian's data: {e}")

    # Pre-phase: Login and fetch REAL current squad with selling prices
    # The public API shows the squad at the last GW deadline, but we need
    # the actual current state including any pending transfers.
    # This session stays alive and is reused for executing transfers later.
    print(c_debug("\n🔐 Authenticating to get real-time squad state..."))
    fpl_actions = FPLActions(team_id)
    logged_in = False
    squad_budget_info = None
    try:
        logged_in = await fpl_actions.login()
        if logged_in:
            squad_budget_info = await fpl_actions.get_squad_with_budget()
            if squad_budget_info and len(squad_budget_info["squad_ids"]) == 15:
                real_squad = squad_budget_info["squad_ids"]
                if set(real_squad) != set(squad_ids):
                    print(c_success("✓ Updated squad with pending transfers"))
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
                    print(c_success("✓ Squad is up to date"))
            else:
                print(c_warn("⚠️  Could not fetch real squad, using cached data"))
                squad_budget_info = None
        else:
            print(c_warn("⚠️  Login failed, using cached squad data"))
    except Exception as e:
        print(c_warn(f"⚠️  Auth error: {e}, using cached squad data"))

    # Scan for budget-compatible top managers
    # Use MARKET prices for the budget since squad costs are calculated with market prices.
    # Selling prices are lower, so using them would filter out everyone.
    budget_candidates = None
    player_prices = {p.id: p.price for p in fpl.get_all_players()}
    scan_budget = sum(player_prices.get(pid, 0.0) for pid in squad_ids)
    if squad_budget_info:
        # Add bank to market-price squad value
        scan_budget += squad_budget_info.get("bank", 0.0)
    if scan_budget < 50.0:
        scan_budget = 100.0
    print(c_debug(f"\n  Scan budget: £{scan_budget:.1f}m (market prices + bank)"))
    try:
        budget_candidates = await find_budget_mirror_targets(
            current_gw=current_gw,
            budget=scan_budget,
            user_squad=squad_ids,
            player_prices=player_prices,
            num_managers=500,
            max_candidates=6,
        )
    except Exception as e:
        print(c_warn(f"  ⚠️  Budget mirror scan failed: {e}"))

    try:
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
            # Phase 1-FH: Free Hit squad builder with mirror options
            chosen_transfers, captain_squad_ids = prompt_free_hit_squad(
                recommender, fpl, squad_ids,
                ian_squad=ian_squad, ian_captain=ian_captain,
                budget_info=squad_budget_info,
                budget_candidates=budget_candidates,
            )
            if chosen_transfers:
                # Build a chip_to_play object for review phase
                chip_to_play = selected_chip
            else:
                # User skipped — no chip
                selected_chip = None

        elif selected_chip == "wildcard":
            # Wildcard uses same Free Hit flow with mirror options
            chosen_transfers, captain_squad_ids = prompt_free_hit_squad(
                recommender, fpl, squad_ids,
                ian_squad=ian_squad, ian_captain=ian_captain,
                budget_info=squad_budget_info,
                budget_candidates=budget_candidates,
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

        # Phase 3: Lineup (starting 11 + bench order)
        lineup_result = prompt_lineup(
            fpl, recommender.scorer, captain_squad_ids,
            captain_id=captain_id, vc_id=vc_id,
        )

        # Phase 4: Review and execute (reuses the existing authenticated session)
        if not logged_in:
            # Initial login failed — try once more before executing
            print("\n🔐 Re-authenticating for execution...")
            logged_in = await fpl_actions.login()

        if logged_in:
            await prompt_review_and_execute(
                fpl_actions, chosen_transfers,
                captain_id, vc_id, chip_to_play, current_gw,
                lineup=lineup_result,
            )
        else:
            print(c_error("❌ Cannot execute — login failed"))

    except UserCancelled:
        print("\n\n  Cancelled. No changes were made.")

    finally:
        # Clean up the session
        await fpl_actions.close()


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

    # Print version and model info
    print(c_header(f"\n{'='*60}"))
    print(c_header(f"  FPL Comeback Assistant v{VERSION}"))
    print(f"  Model: {c_value(MODEL_NAME)}")
    print(f"  {MODEL_DESCRIPTION}")
    print(c_header(f"{'='*60}"))

    print(c_debug("\n🔄 Fetching FPL data..."))

    # Initialize data fetcher
    fpl = FPLDataFetcher()

    # Clear cache if requested
    if args.no_cache:
        fpl.cache.clear()

    # Load all data
    await fpl.load_all_data()

    current_gw = fpl.get_current_gameweek()
    print(c_success(f"✓ Data loaded for GW{current_gw}"))

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
        print(c_debug(f"🔄 Fetching team {args.team_id}..."))
        squad_picks = await fetch_team_squad(fpl, args.team_id)
        team_info = await fetch_team_info(fpl, args.team_id)
        if squad_picks:
            squad_ids = [p["element"] for p in squad_picks]
            print(c_success(f"✓ Found {len(squad_ids)} players in squad"))
            print(c_debug(f"   [debug] squad IDs (from GW picks): {squad_ids}"))
            if team_info:
                name = f"{team_info.get('player_first_name', '')} {team_info.get('player_last_name', '')}".strip()
                team_name = team_info.get('name', 'Unknown')
                print(c_success(f"✓ Manager: {name} | Team: {team_name}"))
        else:
            print("⚠️  Could not fetch team - using general recommendations")

    # Get available chips
    available_chips = get_available_chips(args.chips)

    # Run league spy for auto/default modes when team_id is available
    league_intel = None
    is_auto_or_default = not any([
        args.evaluate, args.compare, args.backtest, args.league_spy,
        args.quiet, args.differentials, args.fixtures, args.top_players,
        args.chip_strategy, args.my_team, args.lineup,
    ])
    if args.team_id and is_auto_or_default:
        try:
            print(c_debug("🔄 Scanning rival squads..."))
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

    # Mirror mode - copy Ian Foster's squad
    if args.mirror:
        if not squad_ids:
            print("❌ --team-id required for mirror mode")
            return

        analysis = await analyze_mirror(
            your_squad=squad_ids,
            current_gw=current_gw,
            free_hit_threshold=args.mirror_threshold,
        )

        if analysis:
            # Build player names map
            player_names = {p.id: p.web_name for p in fpl.get_all_players()}
            print(format_mirror_analysis(analysis, player_names))

            # Show Ian's weekly moves
            print("  IAN'S MOVES THIS GAMEWEEK:")
            moves = await get_ian_weekly_moves(current_gw)
            if moves:
                if moves['transfers_in']:
                    print("    Transfers IN:")
                    for pid in moves['transfers_in']:
                        name = player_names.get(pid, f"Player {pid}")
                        print(f"      + {name}")
                    print("    Transfers OUT:")
                    for pid in moves['transfers_out']:
                        name = player_names.get(pid, f"Player {pid}")
                        print(f"      - {name}")
                else:
                    print("    No transfers made")

                if moves['chip']:
                    print(f"    Chip: {moves['chip'].upper()}")
            print()
        return

    # Generate and display recommendations
    if args.evaluate:
        evaluator = ModelEvaluator(fpl)
        result = await evaluator.evaluate_and_save(args.evaluate, window=args.window)
        print(evaluator.format_result(result, window=args.window))
        return

    if args.league_eval:
        from analysis.evaluator import LeagueEvaluator, WEIGHTS_FILE
        import json

        if not args.team_id:
            print("❌ --team-id required for league evaluation")
            return

        # Build GW range
        gw_range = list(range(args.league_eval - args.window + 1, args.league_eval + 1))

        league_eval = LeagueEvaluator(fpl, args.team_id)
        result = await league_eval.evaluate_league(gw_range)

        # Get model correlation for comparison
        model_corr = 0.0
        ownership_corr = 0.0
        if WEIGHTS_FILE.exists():
            with open(WEIGHTS_FILE) as f:
                history = json.load(f)
            if history:
                # Find matching window entry
                key = f"{args.league_eval}-{args.window}gw" if args.window > 1 else str(args.league_eval)
                if key in history:
                    model_corr = history[key].get("overall_correlation", 0)
                    ownership_corr = history[key].get("correlations", {}).get("ownership", 0)

        print(league_eval.format_league_result(result, model_corr, ownership_corr))
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

    elif args.lineup:
        # Standalone lineup setter
        if not args.team_id:
            print(c_error("❌ --team-id required for lineup mode"))
            return

        if not squad_ids:
            print(c_error("❌ Could not fetch squad"))
            return

        fpl_actions = FPLActions(args.team_id)
        try:
            print(c_debug("\n🔐 Logging in to set lineup..."))
            if not await fpl_actions.login():
                print(c_error("❌ Login failed — cannot set lineup"))
                return

            # Use authenticated squad if possible (includes pending transfers)
            auth_squad = await fpl_actions.get_current_squad()
            if auth_squad:
                squad_ids = auth_squad

            # Get current captain/vc from picks
            cap_id = vc_id = None
            if squad_picks:
                for pk in squad_picks:
                    if pk.get("is_captain"):
                        cap_id = pk["element"]
                    if pk.get("is_vice_captain"):
                        vc_id = pk["element"]

            result = prompt_lineup(fpl, scorer, squad_ids, captain_id=cap_id, vc_id=vc_id)
            if result:
                starting, bench, cap_id, vc_id = result
                await fpl_actions.set_lineup(starting, bench, cap_id, vc_id)
        except UserCancelled:
            print("\n  Cancelled. No changes were made.")
        finally:
            await fpl_actions.close()

    elif args.auto:
        # Interactive auto-pilot mode
        if not args.team_id:
            print(c_error("❌ --team-id required for auto mode"))
            return

        if not squad_ids:
            print(c_error("❌ Could not fetch squad"))
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
