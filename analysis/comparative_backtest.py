"""
Comparative backtest - tests our model against baseline models.
Each model uses only pre-GW data (blind test).
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp

import config
from data.fpl_api import FPLDataFetcher, Player
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.player_scorer import compute_weighted_score

BASE_URL = "https://fantasy.premierleague.com/api"


@dataclass
class ModelScore:
    """Score from a single model for a single GW"""
    model_name: str
    gameweek: int
    top10_avg_pts: float  # Avg actual pts of model's top 10 picks
    top20_overlap: int  # How many of model's top 20 were in actual top 20
    captain_pts: int  # Actual pts of model's #1 pick
    captain_rank: int  # Where the captain actually ranked
    rank_correlation: float  # Spearman vs actual


@dataclass
class ComparisonResult:
    """Full comparison across models and gameweeks"""
    gameweeks: List[int]
    scores: Dict[str, List[ModelScore]]  # model_name -> list of GW scores


class ComparativeBacktester:
    """Runs multiple models on the same GWs for comparison"""

    def __init__(self, fpl_data: FPLDataFetcher):
        self.fpl = fpl_data
        self._gw_history: Dict[int, List[Dict]] = {}

    async def _fetch_histories(self, player_ids: List[int]) -> None:
        batch_size = 20
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(player_ids), batch_size):
                batch = player_ids[i:i + batch_size]
                tasks = []
                for pid in batch:
                    if pid not in self._gw_history:
                        tasks.append(self._fetch_one(session, pid))
                if tasks:
                    await asyncio.gather(*tasks)
                if i + batch_size < len(player_ids):
                    await asyncio.sleep(0.5)

    async def _fetch_one(self, session, pid):
        url = f"{BASE_URL}/element-summary/{pid}/"
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._gw_history[pid] = data.get("history", [])
        except Exception:
            pass

    def _get_pre_gw_data(self, player_id: int, target_gw: int) -> List[Dict]:
        return [h for h in self._gw_history.get(player_id, []) if h["round"] < target_gw and h["minutes"] > 0]

    def _get_actual_pts(self, player_id: int, gw: int) -> int:
        for h in self._gw_history.get(player_id, []):
            if h["round"] == gw:
                return h["total_points"]
        return 0

    def _spearman(self, x: List[float], y: List[float]) -> float:
        n = len(x)
        if n < 3:
            return 0.0
        def rank(arr):
            s = sorted(range(n), key=lambda i: arr[i], reverse=True)
            r = [0.0] * n
            for rk, idx in enumerate(s, 1):
                r[idx] = float(rk)
            return r
        rx, ry = rank(x), rank(y)
        d_sq = sum((rx[i] - ry[i]) ** 2 for i in range(n))
        return 1 - (6 * d_sq) / (n * (n ** 2 - 1))

    # ---- MODEL 1: Our model (weighted composite) ----
    def _score_ours(self, player: Player, pre_gw: List[Dict], fixture_analyzer: FixtureAnalyzer) -> float:
        gp = len(pre_gw)
        total_min = sum(h["minutes"] for h in pre_gw)
        if total_min < 90:
            return 0

        recent = pre_gw[-5:]
        form = min(10, sum(h["total_points"] for h in recent) / len(recent))

        xgi = sum(float(h.get("expected_goal_involvements", 0) or 0) for h in pre_gw)
        xgi_p90 = (xgi / total_min) * 90
        mult = config.XGI_POSITION_MULTIPLIERS.get(player.position, 12.5)
        xgi_score = min(10, xgi_p90 * mult)

        fixture = fixture_analyzer.get_fixture_ease_score(player.team_id, player.position)

        total_pts = sum(h["total_points"] for h in pre_gw)
        price = pre_gw[-1]["value"] / 10
        value = min(10, (total_pts / price) / 1.5) if price > 0 else 0

        ict_total = sum(float(h.get("influence", 0) or 0) + float(h.get("creativity", 0) or 0) + float(h.get("threat", 0) or 0) for h in pre_gw)
        ict = min(10, (ict_total / gp) / 50)

        avg_min = total_min / gp
        minutes = min(10, avg_min / 9)
        starts = sum(1 for h in pre_gw if h["minutes"] >= 60)
        if gp > 3:
            sr = starts / gp
            if sr >= 0.9:
                minutes = min(10, minutes + 1.0)
            elif sr < 0.5:
                minutes *= 0.7

        factors = {
            "form": form,
            "xgi_per_90": xgi_score,
            "fixture_ease": fixture,
            "value_score": value,
            "ict_index": ict,
            "minutes_security": minutes,
        }
        bonuses = {
            "bonus_magnet": min(1.0, sum(h.get("bonus", 0) for h in pre_gw) / gp / 1.5),
        }

        return compute_weighted_score(factors, config.SCORING_WEIGHTS, bonuses)

    # ---- MODEL 2: Last 5 average (simple baseline) ----
    def _score_last5avg(self, pre_gw: List[Dict]) -> float:
        recent = pre_gw[-5:]
        if not recent:
            return 0
        return sum(h["total_points"] for h in recent) / len(recent)

    # ---- MODEL 3: Season average (simplest baseline) ----
    def _score_season_avg(self, pre_gw: List[Dict]) -> float:
        if not pre_gw:
            return 0
        return sum(h["total_points"] for h in pre_gw) / len(pre_gw)

    # ---- MODEL 4: xGI-only model ----
    def _score_xgi_only(self, player: Player, pre_gw: List[Dict]) -> float:
        total_min = sum(h["minutes"] for h in pre_gw)
        if total_min < 90:
            return 0
        xgi = sum(float(h.get("expected_goal_involvements", 0) or 0) for h in pre_gw)
        return (xgi / total_min) * 90

    # ---- MODEL 5: Ownership-based (wisdom of crowds) ----
    def _score_ownership(self, player: Player) -> float:
        return player.selected_by_percent

    def _evaluate_model(
        self,
        model_name: str,
        scores: List[Tuple[int, float]],  # (player_id, score)
        target_gw: int,
        actual_ranked: List[Tuple[int, int]],  # (player_id, actual_pts) sorted desc
    ) -> ModelScore:
        scores.sort(key=lambda x: x[1], reverse=True)

        actual_top20_ids = set(pid for pid, _ in actual_ranked[:20])
        pred_top20_ids = set(pid for pid, _ in scores[:20])
        overlap = len(pred_top20_ids & actual_top20_ids)

        top10_ids = [pid for pid, _ in scores[:10]]
        top10_avg = sum(self._get_actual_pts(pid, target_gw) for pid in top10_ids) / 10

        captain_id = scores[0][0] if scores else None
        captain_pts = self._get_actual_pts(captain_id, target_gw) if captain_id else 0
        captain_rank = 1
        for pid, pts in actual_ranked:
            if pid == captain_id:
                break
            captain_rank += 1

        pred_scores = [s for _, s in scores[:100]]
        actual_pts = [float(self._get_actual_pts(pid, target_gw)) for pid, _ in scores[:100]]
        rank_corr = self._spearman(pred_scores, actual_pts)

        return ModelScore(
            model_name=model_name,
            gameweek=target_gw,
            top10_avg_pts=top10_avg,
            top20_overlap=overlap,
            captain_pts=captain_pts,
            captain_rank=captain_rank,
            rank_correlation=rank_corr,
        )

    async def run_comparison(self, gameweeks: List[int]) -> ComparisonResult:
        """Run all models on given gameweeks"""
        players = [p for p in self.fpl.get_all_players() if p.minutes > 0]
        print(f"   Fetching history for {len(players)} players...")
        await self._fetch_histories([p.id for p in players])

        fixture_analyzer = FixtureAnalyzer(self.fpl)

        all_scores: Dict[str, List[ModelScore]] = {
            "Our Model": [],
            "Last 5 Avg": [],
            "Season Avg": [],
            "xGI Only": [],
            "Ownership": [],
        }

        for gw in gameweeks:
            print(f"\n   Testing GW{gw}...")

            # Get actual results
            actual_ranked = []
            for p in players:
                pts = self._get_actual_pts(p.id, gw)
                actual_ranked.append((p.id, pts))
            actual_ranked.sort(key=lambda x: x[1], reverse=True)

            # Score with each model
            model_scores = {name: [] for name in all_scores}

            for p in players:
                pre = self._get_pre_gw_data(p.id, gw)
                if len(pre) < 3:
                    continue

                model_scores["Our Model"].append((p.id, self._score_ours(p, pre, fixture_analyzer)))
                model_scores["Last 5 Avg"].append((p.id, self._score_last5avg(pre)))
                model_scores["Season Avg"].append((p.id, self._score_season_avg(pre)))
                model_scores["xGI Only"].append((p.id, self._score_xgi_only(p, pre)))
                model_scores["Ownership"].append((p.id, self._score_ownership(p)))

            for name, scores in model_scores.items():
                result = self._evaluate_model(name, scores, gw, actual_ranked)
                all_scores[name].append(result)

        return ComparisonResult(gameweeks=gameweeks, scores=all_scores)

    def format_comparison(self, result: ComparisonResult) -> str:
        lines = [
            f"\n{'=' * 75}",
            f"  COMPARATIVE BLIND TEST: GW{result.gameweeks[0]}-{result.gameweeks[-1]}",
            f"{'=' * 75}",
        ]

        # Per-GW breakdown
        for gw in result.gameweeks:
            lines.append(f"\n  GW{gw}:")
            lines.append(f"  {'Model':<15} {'Top10 Avg':>9} {'Top20 Hit':>9} {'Capt Pts':>9} {'Capt Rank':>10} {'Rank Corr':>10}")
            lines.append(f"  {'-'*64}")

            for name, scores in result.scores.items():
                gw_score = [s for s in scores if s.gameweek == gw]
                if gw_score:
                    s = gw_score[0]
                    lines.append(
                        f"  {name:<15} {s.top10_avg_pts:>8.1f}  {s.top20_overlap:>7}/20  "
                        f"{s.captain_pts:>8}  {s.captain_rank:>9}  {s.rank_correlation:>9.3f}"
                    )

        # Summary averages
        lines.append(f"\n{'=' * 75}")
        lines.append(f"  AVERAGES ACROSS {len(result.gameweeks)} GAMEWEEKS")
        lines.append(f"{'=' * 75}")
        lines.append(f"  {'Model':<15} {'Top10 Avg':>9} {'Top20 Hit':>9} {'Capt Pts':>9} {'Capt Rank':>10} {'Rank Corr':>10}")
        lines.append(f"  {'-'*64}")

        for name, scores in result.scores.items():
            n = len(scores)
            if n == 0:
                continue
            avg_top10 = sum(s.top10_avg_pts for s in scores) / n
            avg_overlap = sum(s.top20_overlap for s in scores) / n
            avg_capt = sum(s.captain_pts for s in scores) / n
            avg_rank = sum(s.captain_rank for s in scores) / n
            avg_corr = sum(s.rank_correlation for s in scores) / n

            lines.append(
                f"  {name:<15} {avg_top10:>8.1f}  {avg_overlap:>7.1f}/20  "
                f"{avg_capt:>8.1f}  {avg_rank:>9.0f}  {avg_corr:>9.3f}"
            )

        # Winner per metric
        lines.append(f"\n  WINNERS:")
        metrics = [
            ("Top 10 Avg Pts", lambda scores: sum(s.top10_avg_pts for s in scores) / len(scores), True),
            ("Top 20 Overlap", lambda scores: sum(s.top20_overlap for s in scores) / len(scores), True),
            ("Captain Pts", lambda scores: sum(s.captain_pts for s in scores) / len(scores), True),
            ("Captain Rank", lambda scores: sum(s.captain_rank for s in scores) / len(scores), False),
            ("Rank Correlation", lambda scores: sum(s.rank_correlation for s in scores) / len(scores), True),
        ]

        for metric_name, fn, higher_better in metrics:
            best_name = None
            best_val = None
            for name, scores in result.scores.items():
                if not scores:
                    continue
                val = fn(scores)
                if best_val is None or (higher_better and val > best_val) or (not higher_better and val < best_val):
                    best_val = val
                    best_name = name
            lines.append(f"    {metric_name:<20} → {best_name}")

        return "\n".join(lines)
