"""
Model evaluator and weight auto-tuner.
Runs after each GW to improve the scoring model.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncio
import aiohttp
import numpy as np

import config
from data.fpl_api import FPLDataFetcher, Player
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.player_scorer import compute_weighted_score

WEIGHTS_FILE = Path("weights_history.json")
LEAGUE_EVAL_FILE = Path("league_evaluation.json")


@dataclass
class FactorCorrelation:
    """Correlation between a scoring factor and actual GW points"""
    name: str
    correlation: float
    current_weight: float
    suggested_weight: float
    direction: str  # "up", "down", "hold"


@dataclass
class EvaluationResult:
    """Result of evaluating the model on a completed GW"""
    gameweek: int
    factor_correlations: List[FactorCorrelation]
    old_weights: Dict[str, float]
    new_weights: Dict[str, float]
    model_rank_corr: float  # Overall Spearman correlation
    ownership_baseline_corr: float = 0.0  # Ownership-only baseline for comparison


class ModelEvaluator:
    """Evaluates model performance and adjusts weights"""

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self, fpl_data: FPLDataFetcher):
        self.fpl = fpl_data
        self._gw_history: Dict[int, List[Dict]] = {}

    async def _fetch_player_gw_history(self, session: aiohttp.ClientSession, player_id: int) -> List[Dict]:
        if player_id in self._gw_history:
            return self._gw_history[player_id]
        url = f"{self.BASE_URL}/element-summary/{player_id}/"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                history = data.get("history", [])
                self._gw_history[player_id] = history
                return history
        except Exception:
            return []

    async def _fetch_histories(self, player_ids: List[int]) -> None:
        batch_size = 20
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(player_ids), batch_size):
                batch = player_ids[i:i + batch_size]
                await asyncio.gather(*[self._fetch_player_gw_history(session, pid) for pid in batch])
                if i + batch_size < len(player_ids):
                    await asyncio.sleep(0.5)

    def _get_pre_gw_factors(self, player: Player, target_gw: int, fixture_analyzer: FixtureAnalyzer) -> Optional[Dict[str, float]]:
        """Compute each scoring factor for a player using only pre-GW data.
        Returns the factor set matching the live model (position-aware)."""
        history = self._gw_history.get(player.id, [])
        pre_gw = [h for h in history if h["round"] < target_gw and h["minutes"] > 0]

        if len(pre_gw) < 3:
            return None

        gp = len(pre_gw)
        total_minutes = sum(h["minutes"] for h in pre_gw)

        if total_minutes < 90:
            return None

        nineties = total_minutes / 90

        # Form: avg points last 5
        recent = pre_gw[-5:]
        form = sum(h["total_points"] for h in recent) / len(recent)
        form_score = min(10, form)

        # xGI per 90
        total_xgi = sum(float(h.get("expected_goal_involvements", 0) or 0) for h in pre_gw)
        xgi_per_90 = (total_xgi / total_minutes) * 90
        mult = config.XGI_POSITION_MULTIPLIERS.get(player.position, 12.5)
        xgi_score = min(10, xgi_per_90 * mult)

        # Fixture ease - use TARGET GW fixtures, not current API state
        # This fixes the backtest leak where we were using future fixture data
        fixture_score = fixture_analyzer.get_fixture_ease_for_gw(player.team_id, target_gw, player.position)

        # Value
        total_pts = sum(h["total_points"] for h in pre_gw)
        price = pre_gw[-1]["value"] / 10 if pre_gw else player.price
        value_score = min(10, (total_pts / price) / 1.5) if price > 0 else 0

        # Minutes
        avg_min = total_minutes / gp
        starts = sum(1 for h in pre_gw if h["minutes"] >= 60)
        minutes_score = min(10, avg_min / 9)
        if gp > 3:
            sr = starts / gp
            if sr >= 0.9:
                minutes_score = min(10, minutes_score + 1.0)
            elif sr < 0.5:
                minutes_score *= 0.7

        # ep_next proxy
        ep_next_score = min(10, form * 1.5)

        # Defensive score (GKP/DEF only)
        defensive_score = 0.0
        if player.position in ("GKP", "DEF") and total_minutes >= 180:
            total_cs = sum(h.get("clean_sheets", 0) for h in pre_gw)
            total_gc = sum(h.get("goals_conceded", 0) for h in pre_gw)
            if player.position == "GKP":
                total_saves = sum(h.get("saves", 0) for h in pre_gw)
                defensive_score = (total_cs / nineties) * 12.5 + (total_saves / nineties) * 0.6
            else:
                defensive_score = (total_cs / nineties) * 12.5 - (total_gc / nineties) * 0.8
            defensive_score = max(0, min(10, defensive_score))

        # ICT position score
        influence_pg = sum(float(h.get("influence", 0) or 0) for h in pre_gw) / gp
        creativity_pg = sum(float(h.get("creativity", 0) or 0) for h in pre_gw) / gp
        threat_pg = sum(float(h.get("threat", 0) or 0) for h in pre_gw) / gp
        if player.position == "FWD":
            ict_pos_raw = threat_pg * 0.6 + creativity_pg * 0.25 + influence_pg * 0.15
        elif player.position == "MID":
            ict_pos_raw = creativity_pg * 0.4 + threat_pg * 0.4 + influence_pg * 0.2
        elif player.position == "DEF":
            ict_pos_raw = influence_pg * 0.5 + creativity_pg * 0.3 + threat_pg * 0.2
        else:
            ict_pos_raw = influence_pg * 0.8 + creativity_pg * 0.1 + threat_pg * 0.1
        ict_position_score = min(10, ict_pos_raw / 5)

        # Ownership score (wisdom of crowds)
        import math
        ownership = player.selected_by_percent
        if ownership > 0:
            ownership_score = math.log(ownership + 1) / math.log(61) * 10
            ownership_score = min(10, max(0, ownership_score))
        else:
            ownership_score = 0.0

        # Recent points score (hot hand) - use last GW before target
        last_gw_pts = 0
        if pre_gw:
            last_gw_pts = pre_gw[-1].get("total_points", 0)
        recent_pts_score = min(10, last_gw_pts / 1.5)
        # Blend with form
        recent_points_score = recent_pts_score * 0.7 + form_score * 0.3

        # Return factor set matching position weights
        pos_weights = getattr(config, "POSITION_WEIGHTS", {}).get(player.position)
        if pos_weights:
            return {
                "ep_next": ep_next_score,
                "form": form_score,
                "xgi_per_90": xgi_score,
                "fixture_ease": fixture_score,
                "defensive": defensive_score,
                "ict_position": ict_position_score,
                "value_score": value_score,
                "minutes_security": minutes_score,
                "ownership": ownership_score,
                "recent_points": recent_points_score,
            }
        else:
            total_ict = sum(float(h.get("influence", 0) or 0) + float(h.get("creativity", 0) or 0) + float(h.get("threat", 0) or 0) for h in pre_gw)
            ict_score = min(10, (total_ict / gp) / 50)
            return {
                "form": form_score,
                "xgi_per_90": xgi_score,
                "fixture_ease": fixture_score,
                "value_score": value_score,
                "ict_index": ict_score,
                "minutes_security": minutes_score,
            }

    def _get_actual_gw_points(self, player_id: int, gw: int) -> Optional[int]:
        history = self._gw_history.get(player_id, [])
        for h in history:
            if h["round"] == gw:
                return h["total_points"]
        return None

    def _spearman(self, x: List[float], y: List[float]) -> float:
        """Spearman rank correlation"""
        n = len(x)
        if n < 3:
            return 0.0

        # Rank both arrays
        def rank(arr):
            sorted_idx = sorted(range(n), key=lambda i: arr[i], reverse=True)
            ranks = [0.0] * n
            for r, idx in enumerate(sorted_idx, 1):
                ranks[idx] = float(r)
            return ranks

        rx = rank(x)
        ry = rank(y)

        d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
        return 1 - (6 * d_sq) / (n * (n ** 2 - 1))

    async def evaluate_gameweek(self, target_gw: int) -> EvaluationResult:
        """Evaluate model on a completed gameweek and suggest weight adjustments.
        Uses the actual position-specific weights when available."""
        players = [p for p in self.fpl.get_all_players() if p.minutes > 0]
        await self._fetch_histories([p.id for p in players])

        fixture_analyzer = FixtureAnalyzer(self.fpl)

        # Determine which weight system we're evaluating
        # Use the first position's weights as representative for factor names
        sample_pos_weights = getattr(config, "POSITION_WEIGHTS", {}).get("MID")
        if sample_pos_weights:
            current_weights = dict(sample_pos_weights)
        else:
            current_weights = dict(config.SCORING_WEIGHTS)

        # Collect factor scores and actual points for all qualifying players
        factor_names = list(current_weights.keys())
        factor_values: Dict[str, List[float]] = {f: [] for f in factor_names}
        actual_points: List[float] = []
        overall_scores: List[float] = []

        for player in players:
            factors = self._get_pre_gw_factors(player, target_gw, fixture_analyzer)
            actual = self._get_actual_gw_points(player.id, target_gw)

            if factors is None or actual is None:
                continue

            for f in factor_names:
                factor_values[f].append(factors.get(f, 0))
            actual_points.append(float(actual))

            # Overall weighted score using position-specific weights
            pos_weights = getattr(config, "POSITION_WEIGHTS", {}).get(player.position)
            weights = pos_weights if pos_weights else config.SCORING_WEIGHTS
            score = compute_weighted_score(factors, weights)
            overall_scores.append(score)

        # Compute correlation per factor
        correlations = []
        total_corr = 0
        for f in factor_names:
            corr = self._spearman(factor_values[f], actual_points)
            total_corr += abs(corr)
            correlations.append((f, corr))

        # Suggest new weights proportional to absolute correlation
        # Blend: 70% current weights + 30% correlation-based
        new_weights = {}
        if total_corr > 0:
            for f, corr in correlations:
                corr_weight = max(0.02, abs(corr)) / total_corr
                new_weights[f] = round(current_weights[f] * 0.7 + corr_weight * 0.3, 4)
        else:
            new_weights = dict(current_weights)

        # Normalize to sum to 1.0
        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            new_weights = {k: round(v / weight_sum, 4) for k, v in new_weights.items()}

        # Build result
        factor_results = []
        for f, corr in correlations:
            old_w = current_weights[f]
            new_w = new_weights[f]
            if new_w > old_w + 0.005:
                direction = "up"
            elif new_w < old_w - 0.005:
                direction = "down"
            else:
                direction = "hold"

            factor_results.append(FactorCorrelation(
                name=f,
                correlation=corr,
                current_weight=old_w,
                suggested_weight=new_w,
                direction=direction,
            ))

        overall_corr = self._spearman(overall_scores, actual_points)

        # Get ownership baseline correlation
        ownership_corr = 0.0
        for fc in factor_results:
            if fc.name == "ownership":
                ownership_corr = fc.correlation
                break

        return EvaluationResult(
            gameweek=target_gw,
            factor_correlations=factor_results,
            old_weights=current_weights,
            new_weights=new_weights,
            model_rank_corr=overall_corr,
            ownership_baseline_corr=ownership_corr,
        )

    async def evaluate_multi_gw(self, gw_range: List[int]) -> EvaluationResult:
        """Evaluate model across multiple gameweeks for robust correlation.

        Pools all player-GW observations together for a larger sample size,
        giving more statistically reliable factor correlations.
        """
        players = [p for p in self.fpl.get_all_players() if p.minutes > 0]
        await self._fetch_histories([p.id for p in players])

        fixture_analyzer = FixtureAnalyzer(self.fpl)

        # Get weight system
        sample_pos_weights = getattr(config, "POSITION_WEIGHTS", {}).get("MID")
        if sample_pos_weights:
            current_weights = dict(sample_pos_weights)
        else:
            current_weights = dict(config.SCORING_WEIGHTS)

        factor_names = list(current_weights.keys())

        # Pool all observations across all GWs
        all_factor_values: Dict[str, List[float]] = {f: [] for f in factor_names}
        all_actual_points: List[float] = []
        all_overall_scores: List[float] = []
        gw_counts = {gw: 0 for gw in gw_range}

        for target_gw in gw_range:
            for player in players:
                factors = self._get_pre_gw_factors(player, target_gw, fixture_analyzer)
                actual = self._get_actual_gw_points(player.id, target_gw)

                if factors is None or actual is None:
                    continue

                for f in factor_names:
                    all_factor_values[f].append(factors.get(f, 0))
                all_actual_points.append(float(actual))

                pos_weights = getattr(config, "POSITION_WEIGHTS", {}).get(player.position)
                weights = pos_weights if pos_weights else config.SCORING_WEIGHTS
                score = compute_weighted_score(factors, weights)
                all_overall_scores.append(score)
                gw_counts[target_gw] += 1

        print(f"   Pooled {len(all_actual_points)} player-GW observations across {len(gw_range)} GWs")
        for gw in gw_range:
            print(f"      GW{gw}: {gw_counts[gw]} observations")

        # Compute correlations on pooled data
        correlations = []
        total_corr = 0
        for f in factor_names:
            corr = self._spearman(all_factor_values[f], all_actual_points)
            total_corr += abs(corr)
            correlations.append((f, corr))

        # Suggest new weights
        new_weights = {}
        if total_corr > 0:
            for f, corr in correlations:
                corr_weight = max(0.02, abs(corr)) / total_corr
                new_weights[f] = round(current_weights[f] * 0.7 + corr_weight * 0.3, 4)
        else:
            new_weights = dict(current_weights)

        weight_sum = sum(new_weights.values())
        if weight_sum > 0:
            new_weights = {k: round(v / weight_sum, 4) for k, v in new_weights.items()}

        # Build result
        factor_results = []
        for f, corr in correlations:
            old_w = current_weights[f]
            new_w = new_weights[f]
            if new_w > old_w + 0.005:
                direction = "up"
            elif new_w < old_w - 0.005:
                direction = "down"
            else:
                direction = "hold"

            factor_results.append(FactorCorrelation(
                name=f,
                correlation=corr,
                current_weight=old_w,
                suggested_weight=new_w,
                direction=direction,
            ))

        overall_corr = self._spearman(all_overall_scores, all_actual_points)

        # Get ownership baseline correlation
        ownership_corr = 0.0
        for fc in factor_results:
            if fc.name == "ownership":
                ownership_corr = fc.correlation
                break

        # Use the last GW in range as the "gameweek" label
        return EvaluationResult(
            gameweek=max(gw_range),
            factor_correlations=factor_results,
            old_weights=current_weights,
            new_weights=new_weights,
            model_rank_corr=overall_corr,
            ownership_baseline_corr=ownership_corr,
        )

    async def evaluate_and_save(self, target_gw: int, window: int = 1) -> EvaluationResult:
        """Evaluate and persist new weights.

        Args:
            target_gw: The target gameweek (or end of range if window > 1)
            window: Number of GWs to evaluate (default 1 = single GW)
        """
        if window > 1:
            gw_range = list(range(target_gw - window + 1, target_gw + 1))
            result = await self.evaluate_multi_gw(gw_range)
        else:
            result = await self.evaluate_gameweek(target_gw)

        # Load or create history
        history = {}
        if WEIGHTS_FILE.exists():
            with open(WEIGHTS_FILE) as f:
                history = json.load(f)

        key = f"{target_gw}" if window == 1 else f"{target_gw}-{window}gw"
        history[key] = {
            "weights": result.new_weights,
            "correlations": {fc.name: fc.correlation for fc in result.factor_correlations},
            "overall_correlation": result.model_rank_corr,
            "ownership_baseline": result.ownership_baseline_corr,
            "window": window,
        }

        with open(WEIGHTS_FILE, "w") as f:
            json.dump(history, f, indent=2)

        return result

    def load_tuned_weights(self) -> Optional[Dict[str, float]]:
        """Load the most recently tuned weights"""
        if not WEIGHTS_FILE.exists():
            return None

        with open(WEIGHTS_FILE) as f:
            history = json.load(f)

        if not history:
            return None

        latest_gw = max(history.keys(), key=int)
        return history[latest_gw]["weights"]

    def format_result(self, r: EvaluationResult, window: int = 1) -> str:
        if window > 1:
            gw_label = f"GW{r.gameweek - window + 1}-{r.gameweek} ({window} weeks)"
        else:
            gw_label = f"GW{r.gameweek}"

        # Calculate advantage over ownership baseline
        advantage = r.model_rank_corr - r.ownership_baseline_corr
        pct_better = (advantage / r.ownership_baseline_corr * 100) if r.ownership_baseline_corr > 0 else 0

        lines = [
            f"\n{'=' * 60}",
            f"  MODEL EVALUATION - {gw_label}",
            f"{'=' * 60}",
            f"\n  MODEL vs OWNERSHIP BASELINE:",
            f"    Our Model:        {r.model_rank_corr:.3f}",
            f"    Ownership Only:   {r.ownership_baseline_corr:.3f}",
        ]

        if advantage >= 0:
            lines.append(f"    Advantage:        +{advantage:.3f} ({pct_better:+.1f}% better)")
        else:
            lines.append(f"    Advantage:        {advantage:.3f} ({pct_better:.1f}% worse)")

        lines.append(f"\n  Factor Analysis:")

        arrow = {"up": "↑", "down": "↓", "hold": "→"}
        for fc in sorted(r.factor_correlations, key=lambda x: x.correlation, reverse=True):
            a = arrow[fc.direction]
            lines.append(
                f"    {fc.name:20} corr: {fc.correlation:+.3f}  "
                f"weight: {fc.current_weight:.2f} {a} {fc.suggested_weight:.2f}"
            )

        lines.append(f"\n  Weights saved to {WEIGHTS_FILE}")
        return "\n".join(lines)

    def get_model_summary(self) -> Optional[Dict]:
        """Get latest evaluation summary for display in other modes."""
        if not WEIGHTS_FILE.exists():
            return None

        try:
            with open(WEIGHTS_FILE) as f:
                history = json.load(f)

            if not history:
                return None

            # Find the latest entry
            def parse_gw(key: str) -> int:
                if "-" in key:
                    return int(key.split("-")[0])
                return int(key)

            latest_key = max(history.keys(), key=parse_gw)
            entry = history[latest_key]

            return {
                "key": latest_key,
                "overall_correlation": entry.get("overall_correlation", 0),
                "ownership_correlation": entry.get("correlations", {}).get("ownership", 0),
                "window": entry.get("window", 1),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None


class LeagueEvaluator:
    """Evaluates how well a mini-league's collective picks predict actual points."""

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self, fpl_data: FPLDataFetcher, your_team_id: int):
        self.fpl = fpl_data
        self.your_team_id = your_team_id
        self._gw_history: Dict[int, List[Dict]] = {}
        self._rival_picks: Dict[int, Dict[int, List[int]]] = {}  # team_id -> {gw -> [player_ids]}

    async def _fetch_json(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Fetch JSON with retry logic"""
        for attempt in range(3):
            try:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    if resp.status in (429, 403):
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
            except Exception:
                await asyncio.sleep(1)
        return {}

    async def _fetch_player_gw_history(self, session: aiohttp.ClientSession, player_id: int) -> List[Dict]:
        if player_id in self._gw_history:
            return self._gw_history[player_id]
        url = f"{self.BASE_URL}/element-summary/{player_id}/"
        data = await self._fetch_json(session, url)
        history = data.get("history", [])
        self._gw_history[player_id] = history
        return history

    async def _fetch_rival_picks_for_gw(self, session: aiohttp.ClientSession, team_id: int, gw: int) -> List[int]:
        """Fetch a rival's squad for a specific GW"""
        if team_id in self._rival_picks and gw in self._rival_picks[team_id]:
            return self._rival_picks[team_id][gw]

        url = f"{self.BASE_URL}/entry/{team_id}/event/{gw}/picks/"
        data = await self._fetch_json(session, url)

        if not data or "picks" not in data:
            return []

        player_ids = [p["element"] for p in data["picks"]]

        if team_id not in self._rival_picks:
            self._rival_picks[team_id] = {}
        self._rival_picks[team_id][gw] = player_ids

        return player_ids

    async def _find_league_rivals(self, session: aiohttp.ClientSession) -> List[int]:
        """Find all rival team IDs in the user's mini-league"""
        # Get user's leagues
        entry_url = f"{self.BASE_URL}/entry/{self.your_team_id}/"
        entry = await self._fetch_json(session, entry_url)

        leagues = entry.get("leagues", {}).get("classic", [])
        private_leagues = [l for l in leagues if l.get("league_type") == "x"]

        if not private_leagues:
            private_leagues = [
                l for l in leagues
                if l.get("entry_rank", 0) > 0 and l.get("entry_rank", 999999) < 50
            ]

        if not private_leagues:
            return []

        # Pick smallest league
        private_leagues.sort(key=lambda l: l.get("entry_rank", 999999))
        league_id = private_leagues[0]["id"]

        # Get standings
        standings_url = f"{self.BASE_URL}/leagues-classic/{league_id}/standings/"
        standings = await self._fetch_json(session, standings_url)

        results = standings.get("standings", {}).get("results", [])
        rival_ids = [r["entry"] for r in results if r["entry"] != self.your_team_id]

        return rival_ids

    def _spearman(self, x: List[float], y: List[float]) -> float:
        """Spearman rank correlation"""
        n = len(x)
        if n < 3:
            return 0.0

        def rank(arr):
            sorted_idx = sorted(range(n), key=lambda i: arr[i], reverse=True)
            ranks = [0.0] * n
            for r, idx in enumerate(sorted_idx, 1):
                ranks[idx] = float(r)
            return ranks

        rx = rank(x)
        ry = rank(y)

        d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
        return 1 - (6 * d_sq) / (n * (n ** 2 - 1))

    async def evaluate_league(self, gw_range: List[int]) -> Dict:
        """Evaluate how well the league's collective ownership predicts actual points.

        For each player-GW, we calculate:
        - League ownership % (what % of rivals owned this player that week)
        - Actual points scored that week

        Then we correlate these to see if the league's wisdom predicts points.
        """
        print(f"\n🔍 Evaluating league correlation over GW{min(gw_range)}-{max(gw_range)}...")

        async with aiohttp.ClientSession() as session:
            # Find rivals
            rival_ids = await self._find_league_rivals(session)
            if not rival_ids:
                print("   Could not find mini-league rivals")
                return {}

            print(f"   Found {len(rival_ids)} rivals in your league")

            # Fetch all player histories
            players = [p for p in self.fpl.get_all_players() if p.minutes > 0]
            print(f"   Fetching history for {len(players)} players...")

            batch_size = 20
            for i in range(0, len(players), batch_size):
                batch = players[i:i + batch_size]
                await asyncio.gather(*[
                    self._fetch_player_gw_history(session, p.id) for p in batch
                ])
                if i + batch_size < len(players):
                    await asyncio.sleep(0.3)

            # Fetch rival picks for each GW
            print(f"   Fetching rival picks for {len(gw_range)} gameweeks...")
            for gw in gw_range:
                for rival_id in rival_ids:
                    await self._fetch_rival_picks_for_gw(session, rival_id, gw)
                    await asyncio.sleep(0.2)
                print(f"      GW{gw} done")

        # Calculate league ownership for each player-GW
        # IMPORTANT: Only count players that are actually owned by at least 1 rival
        # Otherwise correlation is inflated by "owned beats unowned" which is trivial
        league_ownership_scores: List[float] = []
        actual_points: List[float] = []
        global_ownership_for_owned: List[float] = []
        num_rivals = len(rival_ids)

        gw_counts = {gw: 0 for gw in gw_range}
        total_owned = 0

        for gw in gw_range:
            # Count how many rivals own each player this GW
            player_counts: Dict[int, int] = {}
            for rival_id in rival_ids:
                picks = self._rival_picks.get(rival_id, {}).get(gw, [])
                for pid in picks:
                    player_counts[pid] = player_counts.get(pid, 0) + 1

            # For each player OWNED BY AT LEAST 1 RIVAL, record ownership vs points
            for player in players:
                ownership_count = player_counts.get(player.id, 0)
                if ownership_count == 0:
                    continue  # Skip unowned players - they inflate correlation falsely

                history = self._gw_history.get(player.id, [])
                gw_data = next((h for h in history if h["round"] == gw), None)

                if not gw_data:
                    continue

                pts = gw_data.get("total_points", 0)
                ownership_pct = (ownership_count / num_rivals) * 100

                league_ownership_scores.append(ownership_pct)
                actual_points.append(float(pts))
                global_ownership_for_owned.append(player.selected_by_percent)
                gw_counts[gw] += 1
                total_owned += 1

        print(f"   Pooled {len(actual_points)} owned player-GW observations")

        # Calculate correlations - AMONG OWNED PLAYERS ONLY
        league_corr = self._spearman(league_ownership_scores, actual_points)

        # Global ownership correlation for the SAME owned players (fair comparison)
        global_corr = self._spearman(global_ownership_for_owned, actual_points)

        result = {
            "league_correlation": league_corr,
            "global_ownership_correlation": global_corr,
            "observations": len(actual_points),
            "num_rivals": num_rivals,
            "gw_range": [min(gw_range), max(gw_range)],
            "window": len(gw_range),
        }

        # Save result
        with open(LEAGUE_EVAL_FILE, "w") as f:
            json.dump(result, f, indent=2)

        return result

    def format_league_result(self, result: Dict, model_corr: float = 0.0, ownership_corr: float = 0.0) -> str:
        """Format league evaluation results"""
        if not result:
            return "No league evaluation data available"

        league_corr = result.get("league_correlation", 0)
        window = result.get("window", 1)
        obs = result.get("observations", 0)
        rivals = result.get("num_rivals", 0)
        gw_range = result.get("gw_range", [0, 0])

        lines = [
            f"\n{'=' * 60}",
            f"  LEAGUE CORRELATION ANALYSIS - GW{gw_range[0]}-{gw_range[1]} ({window} weeks)",
            f"{'=' * 60}",
            f"\n  Data: {obs:,} player-GW observations from {rivals} rivals",
            f"\n  CORRELATION COMPARISON:",
            f"    Your League:      {league_corr:.3f}  (how well league picks predict points)",
        ]

        if ownership_corr > 0:
            lines.append(f"    Global Ownership: {ownership_corr:.3f}  (wisdom of 11M+ managers)")

        if model_corr > 0:
            lines.append(f"    Our Model:        {model_corr:.3f}  (blended scoring algorithm)")

        # Analysis
        lines.append(f"\n  INTERPRETATION:")

        if model_corr > 0:
            if model_corr > league_corr:
                edge = model_corr - league_corr
                pct = (edge / league_corr * 100) if league_corr > 0 else 0
                lines.append(f"    ✓ Our model beats your league by +{edge:.3f} ({pct:.1f}%)")
                lines.append(f"    → Use model rankings to find edges your rivals miss")
            else:
                gap = league_corr - model_corr
                pct = (gap / model_corr * 100) if model_corr > 0 else 0
                lines.append(f"    ⚠ Your league is {pct:.1f}% better than our model!")
                lines.append(f"    → Tough competition - differentials are crucial")
                lines.append(f"    → Focus on players the league is ignoring")

        if ownership_corr > 0:
            if league_corr > ownership_corr:
                edge = league_corr - ownership_corr
                pct = (edge / ownership_corr * 100) if ownership_corr > 0 else 0
                lines.append(f"\n    Your league ({league_corr:.3f}) beats global ownership ({ownership_corr:.3f})")
                lines.append(f"    These are skilled managers - matching them isn't enough")
            else:
                gap = ownership_corr - league_corr
                pct = (gap / league_corr * 100) if league_corr > 0 else 0
                lines.append(f"\n    Global ownership ({ownership_corr:.3f}) beats your league ({league_corr:.3f})")
                lines.append(f"    Your rivals make suboptimal picks - exploit this!")

        return "\n".join(lines)
