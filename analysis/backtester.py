"""
Blind backtester - tests model predictions against actual results
using only data available before the target gameweek.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp

import config
from data.fpl_api import FPLDataFetcher, Player
from analysis.fixture_analyzer import FixtureAnalyzer
from analysis.player_scorer import compute_weighted_score

# FPL scoring rules per position
CLEAN_SHEET_PTS = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
GOAL_PTS = {"GKP": 6, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_PTS = 3
APPEARANCE_PTS = {(0, 0): 0, (1, 59): 1, (60, 90): 2}


@dataclass
class PlayerGWData:
    """Player stats for a single gameweek"""

    player_id: int
    web_name: str
    team: str
    position: str
    price: float
    gameweek: int
    total_points: int  # Actual points scored this GW
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    bonus: int
    bps: int
    influence: float
    creativity: float
    threat: float
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    selected: int  # ownership count


@dataclass
class PreGWSnapshot:
    """Player state using only data available BEFORE a target gameweek"""

    player_id: int
    web_name: str
    team: str
    team_id: int
    position: str
    price: float
    # Aggregated from GW1 to GW(target-1)
    total_points: int
    total_minutes: int
    total_goals: int
    total_assists: int
    total_clean_sheets: int
    total_bonus: int
    total_bps: int
    total_xg: float
    total_xa: float
    total_xgi: float
    total_influence: float
    total_creativity: float
    total_threat: float
    total_saves: int
    total_goals_conceded: int
    total_transfers_balance: int
    games_played: int
    form: float  # avg points over last 5 GWs played
    starts: int


@dataclass
class BacktestResult:
    """Result of backtesting a single gameweek"""

    target_gw: int
    # Our top N predicted players ranked by score
    predicted_top: List[Tuple[str, float]]  # (web_name, predicted_score)
    # Actual top N players by GW points
    actual_top: List[Tuple[str, int]]  # (web_name, actual_points)
    # Overlap between predicted top N and actual top N
    overlap_count: int
    overlap_pct: float
    # Captain accuracy
    predicted_captain: str
    predicted_captain_actual_pts: int
    best_captain: str
    best_captain_pts: int
    captain_rank: int  # Where our captain pick actually ranked
    # Per-position accuracy
    position_overlap: Dict[str, float]
    # Correlation between predicted rank and actual rank
    rank_correlation: float
    # Summary stats
    predicted_top10_actual_avg: float
    overall_top10_actual_avg: float


class Backtester:
    """Blind backtester using only pre-gameweek data"""

    BASE_URL = "https://fantasy.premierleague.com/api"

    def __init__(self, fpl_data: FPLDataFetcher):
        self.fpl = fpl_data
        self._gw_history: Dict[int, List[Dict]] = {}  # player_id -> gw history

    async def _fetch_player_gw_history(
        self, session: aiohttp.ClientSession, player_id: int
    ) -> List[Dict]:
        """Fetch per-gameweek history for a player"""
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

    async def _fetch_all_histories(self, player_ids: List[int]) -> None:
        """Fetch histories for all players (batched to avoid rate limits)"""
        batch_size = 20
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(player_ids), batch_size):
                batch = player_ids[i : i + batch_size]
                tasks = [self._fetch_player_gw_history(session, pid) for pid in batch]
                await asyncio.gather(*tasks)
                if i + batch_size < len(player_ids):
                    await asyncio.sleep(0.5)  # Rate limit courtesy

    def _build_pre_gw_snapshot(
        self, player: Player, target_gw: int
    ) -> Optional[PreGWSnapshot]:
        """Build player state using only data from GW1 to GW(target-1)"""
        history = self._gw_history.get(player.id, [])
        if not history:
            return None

        # Filter to only gameweeks BEFORE target
        pre_gw = [h for h in history if h["round"] < target_gw]
        if not pre_gw:
            return None

        total_minutes = sum(h["minutes"] for h in pre_gw)
        if total_minutes == 0:
            return None  # Never played

        games_played = sum(1 for h in pre_gw if h["minutes"] > 0)
        starts = sum(1 for h in pre_gw if h["minutes"] >= 60)

        # Form = average points over last 5 appearances
        recent = [h for h in pre_gw if h["minutes"] > 0][-5:]
        form = sum(h["total_points"] for h in recent) / len(recent) if recent else 0

        return PreGWSnapshot(
            player_id=player.id,
            web_name=player.web_name,
            team=player.team,
            team_id=player.team_id,
            position=player.position,
            price=pre_gw[-1]["value"] / 10 if pre_gw else player.price,
            total_points=sum(h["total_points"] for h in pre_gw),
            total_minutes=total_minutes,
            total_goals=sum(h["goals_scored"] for h in pre_gw),
            total_assists=sum(h["assists"] for h in pre_gw),
            total_clean_sheets=sum(h["clean_sheets"] for h in pre_gw),
            total_bonus=sum(h["bonus"] for h in pre_gw),
            total_bps=sum(h["bps"] for h in pre_gw),
            total_xg=sum(float(h.get("expected_goals", 0) or 0) for h in pre_gw),
            total_xa=sum(float(h.get("expected_assists", 0) or 0) for h in pre_gw),
            total_xgi=sum(
                float(h.get("expected_goal_involvements", 0) or 0) for h in pre_gw
            ),
            total_influence=sum(float(h.get("influence", 0) or 0) for h in pre_gw),
            total_creativity=sum(float(h.get("creativity", 0) or 0) for h in pre_gw),
            total_threat=sum(float(h.get("threat", 0) or 0) for h in pre_gw),
            total_saves=sum(h.get("saves", 0) for h in pre_gw),
            total_goals_conceded=sum(h.get("goals_conceded", 0) for h in pre_gw),
            total_transfers_balance=pre_gw[-1].get("transfers_balance", 0)
            if pre_gw
            else 0,
            games_played=games_played,
            form=form,
            starts=starts,
        )

    def _score_snapshot(
        self, snap: PreGWSnapshot, fixture_analyzer: FixtureAnalyzer
    ) -> float:
        """Score a pre-GW snapshot matching the live model as closely as possible.

        Uses POSITION_WEIGHTS (the real model) when available, with all factors
        the live scorer computes. ep_next is unavailable in historical data so
        it's estimated from form as a proxy.
        """
        gp = snap.games_played or 1
        nineties = snap.total_minutes / 90 if snap.total_minutes > 0 else 1

        # Form (0-10)
        form_score = min(10, snap.form)

        # xGI per 90 (0-10), position-aware
        if snap.total_minutes >= 90:
            xgi_per_90 = (snap.total_xgi / snap.total_minutes) * 90
            mult = config.XGI_POSITION_MULTIPLIERS.get(snap.position, 12.5)
            xgi_score = min(10, xgi_per_90 * mult)
        else:
            xgi_score = 0

        # Fixture ease (0-10), position-aware
        fixture_score = fixture_analyzer.get_fixture_ease_score(
            snap.team_id, snap.position
        )

        # Value (0-10)
        if snap.price > 0:
            value_score = min(10, (snap.total_points / snap.price) / 1.5)
        else:
            value_score = 0

        # Minutes security (0-10)
        avg_min = snap.total_minutes / gp
        minutes_score = min(10, avg_min / 9)
        if gp > 3:
            starts_ratio = snap.starts / gp
            if starts_ratio >= 0.9:
                minutes_score = min(10, minutes_score + 1.0)
            elif starts_ratio < 0.5:
                minutes_score *= 0.7

        # ep_next proxy: form * 1.5 (matches live scorer's scaling)
        # The live model uses FPL's ep_next which isn't in historical data
        ep_next_score = min(10, snap.form * 1.5)

        # Defensive score (0-10) — GKP/DEF only
        defensive_score = 0.0
        if snap.position in ("GKP", "DEF") and snap.total_minutes >= 180:
            if snap.position == "GKP":
                saves_per_90 = snap.total_saves / nineties
                defensive_score = (
                    snap.total_clean_sheets / nineties
                ) * 12.5 + saves_per_90 * 0.6
            else:
                gc_per_90 = snap.total_goals_conceded / nineties
                defensive_score = (
                    snap.total_clean_sheets / nineties
                ) * 12.5 - gc_per_90 * 0.8
            defensive_score = max(0, min(10, defensive_score))

        # ICT position score (0-10) — position-aware decomposition
        influence_pg = snap.total_influence / gp
        creativity_pg = snap.total_creativity / gp
        threat_pg = snap.total_threat / gp
        if snap.position == "FWD":
            ict_pos_raw = threat_pg * 0.6 + creativity_pg * 0.25 + influence_pg * 0.15
        elif snap.position == "MID":
            ict_pos_raw = creativity_pg * 0.4 + threat_pg * 0.4 + influence_pg * 0.2
        elif snap.position == "DEF":
            ict_pos_raw = influence_pg * 0.5 + creativity_pg * 0.3 + threat_pg * 0.2
        else:
            ict_pos_raw = influence_pg * 0.8 + creativity_pg * 0.1 + threat_pg * 0.1
        ict_position_score = min(10, ict_pos_raw / 5)

        # Legacy ICT (for SCORING_WEIGHTS fallback)
        ict_total = snap.total_influence + snap.total_creativity + snap.total_threat
        ict_score = min(10, (ict_total / gp) / 50)

        # Use position-specific weights if available (matches live model)
        pos_weights = getattr(config, "POSITION_WEIGHTS", {}).get(snap.position)

        if pos_weights:
            factors = {
                "ep_next": ep_next_score,
                "form": form_score,
                "xgi_per_90": xgi_score,
                "fixture_ease": fixture_score,
                "defensive": defensive_score,
                "ict_position": ict_position_score,
                "value_score": value_score,
                "minutes_security": minutes_score,
            }
            weights = pos_weights
        else:
            factors = {
                "form": form_score,
                "xgi_per_90": xgi_score,
                "fixture_ease": fixture_score,
                "value_score": value_score,
                "ict_index": ict_score,
                "minutes_security": minutes_score,
            }
            weights = config.SCORING_WEIGHTS

        # Bonuses — same as live model (set_piece not available in history)
        bonuses = {
            "bonus_magnet": min(1.0, (snap.total_bonus / gp) / 1.5),
            "transfer_momentum": max(0, min(2.0, snap.total_transfers_balance / 50000)),
        }

        return compute_weighted_score(factors, weights, bonuses)

    def _get_actual_gw_points(self, player_id: int, target_gw: int) -> int:
        """Get actual points for a player in a specific gameweek"""
        history = self._gw_history.get(player_id, [])
        for h in history:
            if h["round"] == target_gw:
                return h["total_points"]
        return 0

    async def run_backtest(
        self,
        target_gw: int,
        top_n: int = 20,
        verbose: bool = True,
    ) -> BacktestResult:
        """Run blind backtest for a single gameweek"""
        if verbose:
            print(f"\n🔬 BACKTESTING GW{target_gw}")
            print(f"   Using only data from GW1-{target_gw - 1}...")

        # Get all players who have meaningful minutes
        all_players = [p for p in self.fpl.get_all_players() if p.minutes > 0]
        player_ids = [p.id for p in all_players]

        if verbose:
            print(f"   Fetching history for {len(player_ids)} players...")

        await self._fetch_all_histories(player_ids)

        # Build pre-GW snapshots
        fixture_analyzer = FixtureAnalyzer(self.fpl)
        scored: List[Tuple[PreGWSnapshot, float]] = []

        for player in all_players:
            snap = self._build_pre_gw_snapshot(player, target_gw)
            if snap and snap.games_played >= 3:  # Need minimum data
                score = self._score_snapshot(snap, fixture_analyzer)
                scored.append((snap, score))

        # Sort by predicted score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Get actual GW results
        actual_results: List[
            Tuple[str, int, int, str]
        ] = []  # (name, actual_pts, player_id, position)
        for player in all_players:
            actual_pts = self._get_actual_gw_points(player.id, target_gw)
            actual_results.append(
                (player.web_name, actual_pts, player.id, player.position)
            )

        actual_results.sort(key=lambda x: x[1], reverse=True)

        # Our predicted top N
        predicted_top = [(s.web_name, score) for s, score in scored[:top_n]]
        predicted_names = set(name for name, _ in predicted_top)

        # Actual top N
        actual_top = [(name, pts) for name, pts, _, _ in actual_results[:top_n]]
        actual_names = set(name for name, _ in actual_top)

        # Overlap
        overlap = predicted_names & actual_names
        overlap_pct = len(overlap) / top_n * 100

        # Captain analysis
        predicted_captain = predicted_top[0][0] if predicted_top else "N/A"
        pred_captain_id = scored[0][0].player_id if scored else None
        pred_captain_actual = (
            self._get_actual_gw_points(pred_captain_id, target_gw)
            if pred_captain_id
            else 0
        )

        best_captain = actual_results[0][0] if actual_results else "N/A"
        best_captain_pts = actual_results[0][1] if actual_results else 0

        # Where did our captain actually rank?
        captain_rank = 1
        for name, pts, pid, _ in actual_results:
            if pid == pred_captain_id:
                break
            captain_rank += 1

        # Per-position overlap
        position_overlap = {}
        for pos in ["GKP", "DEF", "MID", "FWD"]:
            pos_predicted = set(
                s.web_name for s, _ in scored[:top_n] if s.position == pos
            )
            pos_actual = set(
                name for name, _, _, p in actual_results[:top_n] if p == pos
            )
            if pos_predicted and pos_actual:
                pos_overlap = len(pos_predicted & pos_actual)
                pos_total = max(len(pos_predicted), len(pos_actual))
                position_overlap[pos] = (
                    pos_overlap / pos_total * 100 if pos_total > 0 else 0
                )

        # Rank correlation (Spearman's)
        predicted_ids = [s.player_id for s, _ in scored[:100]]
        actual_pts_map = {pid: pts for _, pts, pid, _ in actual_results}

        pred_ranks = {pid: rank for rank, pid in enumerate(predicted_ids, 1)}
        actual_sorted = sorted(
            predicted_ids, key=lambda pid: actual_pts_map.get(pid, 0), reverse=True
        )
        actual_ranks = {pid: rank for rank, pid in enumerate(actual_sorted, 1)}

        n = len(predicted_ids)
        if n > 1:
            d_squared = sum(
                (pred_ranks[pid] - actual_ranks.get(pid, n)) ** 2
                for pid in predicted_ids
            )
            rank_corr = 1 - (6 * d_squared) / (n * (n**2 - 1))
        else:
            rank_corr = 0

        # Avg actual points of our predicted top 10
        pred_top10_ids = [s.player_id for s, _ in scored[:10]]
        pred_top10_actual_avg = (
            sum(self._get_actual_gw_points(pid, target_gw) for pid in pred_top10_ids)
            / 10
        )

        # Avg actual points of overall top 10
        overall_top10_avg = sum(pts for _, pts in actual_top[:10]) / 10

        result = BacktestResult(
            target_gw=target_gw,
            predicted_top=predicted_top,
            actual_top=actual_top,
            overlap_count=len(overlap),
            overlap_pct=overlap_pct,
            predicted_captain=predicted_captain,
            predicted_captain_actual_pts=pred_captain_actual,
            best_captain=best_captain,
            best_captain_pts=best_captain_pts,
            captain_rank=captain_rank,
            position_overlap=position_overlap,
            rank_correlation=rank_corr,
            predicted_top10_actual_avg=pred_top10_actual_avg,
            overall_top10_actual_avg=overall_top10_avg,
        )

        if verbose:
            print(self.format_result(result))

        return result

    async def run_multi_gw_backtest(
        self,
        start_gw: int,
        end_gw: int,
        top_n: int = 20,
    ) -> List[BacktestResult]:
        """Run backtest across multiple gameweeks"""
        print(f"\n{'=' * 60}")
        print(f"  MULTI-GAMEWEEK BACKTEST: GW{start_gw} to GW{end_gw}")
        print(f"{'=' * 60}")

        results = []
        for gw in range(start_gw, end_gw + 1):
            result = await self.run_backtest(gw, top_n=top_n, verbose=True)
            results.append(result)

        # Summary
        print(f"\n{'=' * 60}")
        print(f"  BACKTEST SUMMARY (GW{start_gw}-{end_gw})")
        print(f"{'=' * 60}")

        avg_overlap = sum(r.overlap_pct for r in results) / len(results)
        avg_rank_corr = sum(r.rank_correlation for r in results) / len(results)
        avg_captain_pts = sum(r.predicted_captain_actual_pts for r in results) / len(
            results
        )
        avg_best_captain = sum(r.best_captain_pts for r in results) / len(results)
        avg_pred_top10 = sum(r.predicted_top10_actual_avg for r in results) / len(
            results
        )
        avg_actual_top10 = sum(r.overall_top10_actual_avg for r in results) / len(
            results
        )
        avg_captain_rank = sum(r.captain_rank for r in results) / len(results)

        print(f"\n  Top-{top_n} Overlap:        {avg_overlap:.1f}% avg")
        print(f"  Rank Correlation:        {avg_rank_corr:.3f} avg (Spearman)")
        print(f"  Our Captain Avg Pts:     {avg_captain_pts:.1f}")
        print(f"  Best Captain Avg Pts:    {avg_best_captain:.1f}")
        print(
            f"  Captain Efficiency:      {avg_captain_pts / avg_best_captain * 100:.0f}%"
            if avg_best_captain
            else "  N/A"
        )
        print(f"  Avg Captain Rank:        {avg_captain_rank:.0f}")
        print(f"  Our Top 10 Avg Pts:      {avg_pred_top10:.1f}")
        print(f"  Actual Top 10 Avg Pts:   {avg_actual_top10:.1f}")
        print(
            f"  Selection Efficiency:    {avg_pred_top10 / avg_actual_top10 * 100:.0f}%"
            if avg_actual_top10
            else "  N/A"
        )

        return results

    def format_result(self, r: BacktestResult) -> str:
        """Format a single backtest result"""
        lines = [
            f"\n   --- GW{r.target_gw} RESULTS ---",
            f"   Top-20 Overlap: {r.overlap_count}/20 ({r.overlap_pct:.0f}%)",
            f"   Rank Correlation: {r.rank_correlation:.3f}",
            "",
            f"   Captain: {r.predicted_captain} → {r.predicted_captain_actual_pts} pts "
            f"(ranked #{r.captain_rank} overall)",
            f"   Best:    {r.best_captain} → {r.best_captain_pts} pts",
            "",
            f"   Our Top 10 avg:    {r.predicted_top10_actual_avg:.1f} pts",
            f"   Actual Top 10 avg: {r.overall_top10_actual_avg:.1f} pts",
            "",
            "   PREDICTED TOP 5:          ACTUAL TOP 5:",
        ]

        for i in range(5):
            pred = (
                f"{r.predicted_top[i][0]:15} ({r.predicted_top[i][1]:.1f})"
                if i < len(r.predicted_top)
                else ""
            )
            actual = (
                f"{r.actual_top[i][0]:15} ({r.actual_top[i][1]} pts)"
                if i < len(r.actual_top)
                else ""
            )
            lines.append(f"   {i + 1}. {pred:30} {actual}")

        return "\n".join(lines)
