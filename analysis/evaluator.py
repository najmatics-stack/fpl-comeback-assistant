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

WEIGHTS_FILE = Path("weights_history.json")


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
        """Compute each scoring factor for a player using only pre-GW data"""
        history = self._gw_history.get(player.id, [])
        pre_gw = [h for h in history if h["round"] < target_gw and h["minutes"] > 0]

        if len(pre_gw) < 3:
            return None

        gp = len(pre_gw)
        total_minutes = sum(h["minutes"] for h in pre_gw)

        if total_minutes < 90:
            return None

        # Form: avg points last 5
        recent = pre_gw[-5:]
        form = sum(h["total_points"] for h in recent) / len(recent)
        form_score = min(10, form)

        # xGI per 90
        total_xgi = sum(float(h.get("expected_goal_involvements", 0) or 0) for h in pre_gw)
        xgi_per_90 = (total_xgi / total_minutes) * 90
        mult = {"GKP": 25.0, "DEF": 20.0, "MID": 12.5, "FWD": 11.0}.get(player.position, 12.5)
        xgi_score = min(10, xgi_per_90 * mult)

        # Fixture ease
        fixture_score = fixture_analyzer.get_fixture_ease_score(player.team_id, player.position)

        # Value
        total_pts = sum(h["total_points"] for h in pre_gw)
        price = pre_gw[-1]["value"] / 10 if pre_gw else player.price
        value_score = min(10, (total_pts / price) / 1.5) if price > 0 else 0

        # ICT
        total_ict = sum(float(h.get("influence", 0) or 0) + float(h.get("creativity", 0) or 0) + float(h.get("threat", 0) or 0) for h in pre_gw)
        ict_score = min(10, (total_ict / gp) / 50)

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
        """Evaluate model on a completed gameweek and suggest weight adjustments"""
        players = [p for p in self.fpl.get_all_players() if p.minutes > 0]
        await self._fetch_histories([p.id for p in players])

        fixture_analyzer = FixtureAnalyzer(self.fpl)
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
                factor_values[f].append(factors[f])
            actual_points.append(float(actual))

            # Overall weighted score
            score = sum(factors[f] * current_weights[f] for f in factor_names)
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

        return EvaluationResult(
            gameweek=target_gw,
            factor_correlations=factor_results,
            old_weights=current_weights,
            new_weights=new_weights,
            model_rank_corr=overall_corr,
        )

    async def evaluate_and_save(self, target_gw: int) -> EvaluationResult:
        """Evaluate and persist new weights"""
        result = await self.evaluate_gameweek(target_gw)

        # Load or create history
        history = {}
        if WEIGHTS_FILE.exists():
            with open(WEIGHTS_FILE) as f:
                history = json.load(f)

        history[str(target_gw)] = {
            "weights": result.new_weights,
            "correlations": {fc.name: fc.correlation for fc in result.factor_correlations},
            "overall_correlation": result.model_rank_corr,
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

    def format_result(self, r: EvaluationResult) -> str:
        lines = [
            f"\n{'=' * 60}",
            f"  MODEL EVALUATION - GW{r.gameweek}",
            f"{'=' * 60}",
            f"\n  Overall Rank Correlation: {r.model_rank_corr:.3f}",
            f"\n  Factor Analysis:",
        ]

        arrow = {"up": "↑", "down": "↓", "hold": "→"}
        for fc in sorted(r.factor_correlations, key=lambda x: x.correlation, reverse=True):
            a = arrow[fc.direction]
            lines.append(
                f"    {fc.name:20} corr: {fc.correlation:+.3f}  "
                f"weight: {fc.current_weight:.2f} {a} {fc.suggested_weight:.2f}"
            )

        lines.append(f"\n  Weights saved to {WEIGHTS_FILE}")
        return "\n".join(lines)
