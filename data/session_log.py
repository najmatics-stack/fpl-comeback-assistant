"""
Session logging for FPL Comeback Assistant

Logs each --auto run to a JSON file for post-GW analysis.
Logs are stored in logs/ directory with filename: gw{N}_{timestamp}.json
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGS_DIR = Path(__file__).parent.parent / "logs"


class SessionLog:
    """Captures decisions made during an --auto session for later analysis."""

    def __init__(self, gameweek: int, team_id: int):
        self.gameweek = gameweek
        self.team_id = team_id
        self.timestamp = datetime.now().isoformat()
        self.data: Dict[str, Any] = {
            "gameweek": gameweek,
            "team_id": team_id,
            "timestamp": self.timestamp,
            "model_version": None,
            "chip_used": None,
            "squad_before": [],
            "squad_after": [],
            "transfers": [],
            "captain_options": [],
            "captain_chosen": None,
            "vice_captain_chosen": None,
            "lineup": {"starting": [], "bench": []},
            "mirror_target": None,
            "budget_candidates": [],
            "execution_status": None,
        }

    def set_model_version(self, version: str, name: str):
        self.data["model_version"] = {"version": version, "name": name}

    def set_chip(self, chip: Optional[str]):
        self.data["chip_used"] = chip

    def set_squad_before(self, squad_ids: List[int], player_names: Dict[int, str]):
        self.data["squad_before"] = [
            {"id": pid, "name": player_names.get(pid, f"ID:{pid}")}
            for pid in squad_ids
        ]

    def set_squad_after(self, squad_ids: List[int], player_names: Dict[int, str]):
        self.data["squad_after"] = [
            {"id": pid, "name": player_names.get(pid, f"ID:{pid}")}
            for pid in squad_ids
        ]

    def add_transfer(
        self,
        out_id: int,
        out_name: str,
        in_id: int,
        in_name: str,
        score_gain: float,
    ):
        self.data["transfers"].append(
            {
                "out": {"id": out_id, "name": out_name},
                "in": {"id": in_id, "name": in_name},
                "score_gain": round(score_gain, 2),
            }
        )

    def set_captain_options(
        self,
        options: List[Dict],
    ):
        """
        options: list of {id, name, expected_points, ownership, form, fixture}
        """
        self.data["captain_options"] = options

    def set_captain_chosen(
        self,
        captain_id: int,
        captain_name: str,
        vc_id: int,
        vc_name: str,
    ):
        self.data["captain_chosen"] = {"id": captain_id, "name": captain_name}
        self.data["vice_captain_chosen"] = {"id": vc_id, "name": vc_name}

    def set_lineup(
        self,
        starting: List[int],
        bench: List[int],
        player_names: Dict[int, str],
    ):
        self.data["lineup"]["starting"] = [
            {"id": pid, "name": player_names.get(pid, f"ID:{pid}")}
            for pid in starting
        ]
        self.data["lineup"]["bench"] = [
            {"id": pid, "name": player_names.get(pid, f"ID:{pid}")}
            for pid in bench
        ]

    def set_mirror_target(self, name: str, rank: int, squad_ids: List[int]):
        self.data["mirror_target"] = {
            "name": name,
            "rank": rank,
            "squad_ids": squad_ids,
        }

    def add_budget_candidate(
        self,
        rank: int,
        name: str,
        cost: float,
        fits_budget: bool,
    ):
        self.data["budget_candidates"].append(
            {
                "rank": rank,
                "name": name,
                "cost": round(cost, 1),
                "fits_budget": fits_budget,
            }
        )

    def set_execution_status(self, status: str, details: Optional[str] = None):
        self.data["execution_status"] = {"status": status, "details": details}

    def save(self) -> Path:
        """Save log to file and return the path."""
        LOGS_DIR.mkdir(exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gw{self.gameweek}_{ts}.json"
        filepath = LOGS_DIR / filename

        with open(filepath, "w") as f:
            json.dump(self.data, f, indent=2)

        return filepath

    def format_summary(self) -> str:
        """Return a human-readable summary for terminal output."""
        lines = [
            "",
            "=" * 60,
            f"  SESSION LOG - GW{self.gameweek}",
            "=" * 60,
            f"  Timestamp: {self.timestamp}",
            f"  Chip: {self.data['chip_used'] or 'None'}",
        ]

        if self.data["transfers"]:
            lines.append(f"\n  Transfers ({len(self.data['transfers'])}):")
            for tr in self.data["transfers"]:
                lines.append(
                    f"    {tr['out']['name']} -> {tr['in']['name']} "
                    f"[+{tr['score_gain']:.1f}]"
                )

        if self.data["captain_chosen"]:
            cap = self.data["captain_chosen"]
            vc = self.data["vice_captain_chosen"]
            lines.append(f"\n  Captain: {cap['name']}")
            lines.append(f"  Vice Captain: {vc['name']}")

        if self.data["captain_options"]:
            lines.append("\n  Captain Options (model ranking):")
            for i, opt in enumerate(self.data["captain_options"][:5], 1):
                chosen = " <- CHOSEN" if opt["id"] == self.data["captain_chosen"]["id"] else ""
                lines.append(
                    f"    {i}. {opt['name']:15} "
                    f"exp:{opt.get('expected_points', 0):.1f} "
                    f"own:{opt.get('ownership', 0):.1f}%{chosen}"
                )

        if self.data["mirror_target"]:
            mt = self.data["mirror_target"]
            lines.append(f"\n  Mirror Target: #{mt['rank']} {mt['name']}")

        status = self.data.get("execution_status", {})
        if status:
            lines.append(f"\n  Execution: {status.get('status', 'unknown')}")
            if status.get("details"):
                lines.append(f"    {status['details']}")

        return "\n".join(lines)


def get_latest_log(gameweek: Optional[int] = None) -> Optional[Dict]:
    """Load the most recent log file, optionally filtered by gameweek."""
    if not LOGS_DIR.exists():
        return None

    log_files = sorted(LOGS_DIR.glob("gw*.json"), reverse=True)

    for lf in log_files:
        with open(lf) as f:
            data = json.load(f)
        if gameweek is None or data.get("gameweek") == gameweek:
            return data

    return None


def list_logs() -> List[Dict]:
    """Return metadata for all log files."""
    if not LOGS_DIR.exists():
        return []

    results = []
    for lf in sorted(LOGS_DIR.glob("gw*.json"), reverse=True):
        with open(lf) as f:
            data = json.load(f)
        results.append(
            {
                "file": lf.name,
                "gameweek": data.get("gameweek"),
                "timestamp": data.get("timestamp"),
                "chip": data.get("chip_used"),
                "captain": data.get("captain_chosen", {}).get("name"),
            }
        )
    return results
