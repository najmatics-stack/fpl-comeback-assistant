#!/usr/bin/env python3
"""
FPL Deadline Checker — sends Discord notifications when a GW deadline approaches,
and squad availability alerts when players are flagged.

All notifications are limited to the 24-hour window before deadline:
  - 1st alert: ~24h before deadline  (18h < remaining ≤ 24h)
  - 2nd alert: ~6h before deadline   ( 0h < remaining ≤  6h)

Squad alerts are bundled into deadline notifications (not sent separately).
Outside the 24h window, nothing is sent.

Usage:
  DISCORD_WEBHOOK_URL=<url> python3 check_deadline.py
  python3 check_deadline.py --test   # send test messages immediately
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import aiohttp

import config
from data.discord_notifier import send_deadline_alert, send_squad_alert

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
FPL_MY_TEAM_URL = "https://fantasy.premierleague.com/api/my-team/{team_id}/"
SESSION_FILE = Path.home() / ".fpl_session"
# Only notify within 24h of deadline — two windows max.
# (upper_seconds, lower_seconds, label)
NOTIFICATION_WINDOWS = [
    (24 * 3600, 18 * 3600, "24 hours"),
    (6 * 3600, 0, "6 hours"),
]


async def fetch_bootstrap() -> dict:
    """Fetch raw bootstrap-static data from the FPL API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(FPL_BOOTSTRAP_URL) as resp:
            return await resp.json(content_type=None)


def parse_next_deadline(bootstrap_data: dict) -> dict | None:
    """Extract the next upcoming gameweek deadline from bootstrap data (pure function)."""
    for event in bootstrap_data["events"]:
        if event.get("is_next"):
            return {
                "gw": event["id"],
                "name": event["name"],
                "deadline_epoch": event["deadline_time_epoch"],
                "deadline_str": event["deadline_time"],
            }
    return None


def _load_fpl_session() -> Optional[Dict[str, str]]:
    """Load saved FPL session from ~/.fpl_session if fresh enough."""
    if not SESSION_FILE.exists():
        return None
    try:
        data = json.loads(SESSION_FILE.read_text())
        if time.time() - data.get("saved_at", 0) > 7 * 86400:
            return None
        return data.get("session")
    except Exception:
        return None


async def _fetch_my_team_squad(
    team_id: int,
    session_data: Dict[str, str],
    elements: Dict[int, dict],
) -> Tuple[Optional[dict], bool]:
    """Fetch real current squad via authenticated /my-team/ endpoint.

    Returns (picks_data_dict, True) on success, (None, False) on failure.
    The picks_data_dict has the same shape as the public picks endpoint.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
        ),
        "Referer": "https://fantasy.premierleague.com/",
        "Origin": "https://fantasy.premierleague.com",
        "X-Requested-With": "XMLHttpRequest",
    }

    # Cookies
    cookie_keys = [k for k in session_data if not k.startswith("_token")]
    if cookie_keys:
        headers["Cookie"] = "; ".join(
            f"{k}={session_data[k]}" for k in cookie_keys
        )

    # Bearer token (OAuth2)
    access_token = session_data.get("_token_access")
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    csrf = session_data.get("csrftoken")
    if csrf:
        headers["X-CSRFToken"] = csrf

    url = FPL_MY_TEAM_URL.format(team_id=team_id)
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None, False
            data = await resp.json(content_type=None)

    picks = data.get("picks", [])
    if not picks:
        return None, False

    # /my-team/ picks have: element, position, is_captain, is_vice_captain, etc.
    picks_data = {
        "picks": [
            {
                "element": p["element"],
                "is_captain": p.get("is_captain", False),
                "is_vice_captain": p.get("is_vice_captain", False),
            }
            for p in picks
        ]
    }
    print(f"[squad_check] Using live squad from /my-team/ ({len(picks)} players)")
    return picks_data, True


async def check_squad_availability(
    bootstrap_data: dict,
    team_id: int,
    current_gw: int,
) -> Tuple[list[dict], bool]:
    """
    Check squad players for availability issues.

    Returns (flagged_players, squad_is_live) where squad_is_live indicates
    whether the data came from the authenticated /my-team/ endpoint (True)
    or the public GW-1 picks fallback (False).
    """
    # Build element lookup
    elements = {el["id"]: el for el in bootstrap_data["elements"]}

    # Try authenticated /my-team/ endpoint first (shows real current squad
    # including pending transfers).  Falls back to public GW-1 picks.
    picks_data = None
    squad_is_live = False

    session_data = _load_fpl_session()
    if session_data:
        try:
            picks_data, squad_is_live = await _fetch_my_team_squad(
                team_id, session_data, elements
            )
        except Exception:
            pass

    # Fallback: public GW-1 picks (last confirmed lineup, no pending transfers)
    if not picks_data:
        base_gw = current_gw - 1
        if base_gw >= 1:
            url = FPL_PICKS_URL.format(team_id=team_id, gw=base_gw)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            picks_data = await resp.json(content_type=None)
            except Exception:
                pass

        # Infer likely captain/VC from highest ep_next since the GW-1
        # captain flags are stale and meaningless before the new deadline.
        if picks_data:
            squad_element_ids = {p["element"] for p in picks_data["picks"]}
            likely_captain = max(
                squad_element_ids,
                key=lambda eid: float(
                    elements.get(eid, {}).get("ep_next", 0) or 0
                ),
            )
            remaining_ids = squad_element_ids - {likely_captain}
            likely_vc = (
                max(
                    remaining_ids,
                    key=lambda eid: float(
                        elements.get(eid, {}).get("ep_next", 0) or 0
                    ),
                )
                if remaining_ids
                else None
            )
            picks_data["picks"] = [
                {
                    "element": eid,
                    "is_captain": eid == likely_captain,
                    "is_vice_captain": eid == likely_vc,
                }
                for eid in squad_element_ids
            ]

    if not picks_data:
        print(f"[squad_check] Could not fetch picks for team {team_id}")
        return [], False

    picks = picks_data.get("picks", [])
    flagged = []

    for pick in picks:
        element_id = pick["element"]
        el = elements.get(element_id)
        if not el:
            continue

        status = el.get("status", "a")
        chance = el.get("chance_of_playing_next_round")

        # Flag if not available, or chance <= 50%
        is_flagged = (
            status != "a"
            or (chance is not None and chance <= 50)
        )

        if is_flagged:
            # Position name from element_type
            pos_map = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
            flagged.append({
                "name": el.get("web_name", f"ID:{element_id}"),
                "status": status,
                "chance": chance,
                "news": el.get("news", ""),
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
                "position": pos_map.get(el.get("element_type", 0), "???"),
            })

    return flagged, squad_is_live


async def main() -> None:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL environment variable not set.")
        sys.exit(1)

    team_id = config.TEAM_ID

    # --test flag: send test messages and exit
    if "--test" in sys.argv:
        now = int(datetime.now(timezone.utc).timestamp())
        success_deadline = await send_deadline_alert(
            webhook_url=webhook_url,
            gw_name="Gameweek 99 (TEST)",
            gw_number=99,
            deadline_epoch=now + 6 * 3600,
            seconds_remaining=6 * 3600,
            window_label="6 hours",
        )
        print("✓ Test deadline alert sent." if success_deadline else "✗ Deadline alert failed.")

        success_squad = await send_squad_alert(
            webhook_url=webhook_url,
            gw_name="Gameweek 99 (TEST)",
            flagged_players=[
                {
                    "name": "Haaland",
                    "status": "i",
                    "chance": 0,
                    "news": "Knee injury - expected back in 3 weeks",
                    "is_captain": True,
                    "is_vice_captain": False,
                    "position": "FWD",
                },
                {
                    "name": "Saka",
                    "status": "d",
                    "chance": 25,
                    "news": "Hamstring - 25% chance of playing",
                    "is_captain": False,
                    "is_vice_captain": True,
                    "position": "MID",
                },
            ],
            captain_affected=True,
        )
        print("✓ Test squad alert sent." if success_squad else "✗ Squad alert failed.")
        return

    # Fetch bootstrap data (shared by deadline + squad check)
    print("Fetching FPL bootstrap data...")
    bootstrap_data = await fetch_bootstrap()

    gw_info = parse_next_deadline(bootstrap_data)

    # --now flag: send the real current deadline immediately, ignoring windows
    if "--now" in sys.argv:
        if not gw_info:
            print("No upcoming gameweek found.")
            return
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        seconds_remaining = gw_info["deadline_epoch"] - now_epoch
        hours_remaining = seconds_remaining / 3600
        label = f"{hours_remaining:.0f} hours"

        # Include squad issues in --now output
        flagged, squad_is_live = await check_squad_availability(
            bootstrap_data, team_id, gw_info["gw"]
        )

        success = await send_deadline_alert(
            webhook_url=webhook_url,
            gw_name=gw_info["name"],
            gw_number=gw_info["gw"],
            deadline_epoch=gw_info["deadline_epoch"],
            seconds_remaining=seconds_remaining,
            window_label=label,
        )
        if success and flagged:
            captain_affected = any(p.get("is_captain") for p in flagged)
            await send_squad_alert(
                webhook_url=webhook_url,
                gw_name=gw_info["name"],
                flagged_players=flagged,
                captain_affected=captain_affected,
                squad_is_live=squad_is_live,
            )
        print("✓ Notification sent." if success else "✗ Notification failed.")
        return

    if not gw_info:
        print("No upcoming gameweek found — season may be over.")
        return

    now_epoch = int(datetime.now(timezone.utc).timestamp())
    seconds_remaining = gw_info["deadline_epoch"] - now_epoch
    hours_remaining = seconds_remaining / 3600

    print(
        f"Next GW: {gw_info['name']} | "
        f"Deadline: {gw_info['deadline_str']} | "
        f"Remaining: {hours_remaining:.1f}h"
    )

    if seconds_remaining <= 0:
        print("Deadline has already passed.")
        return

    # Only notify within 24h of deadline — everything else is silent
    for upper, lower, label in NOTIFICATION_WINDOWS:
        if lower < seconds_remaining <= upper:
            print(f"→ In '{label}' window — sending Discord notification...")

            # Check squad availability and bundle with deadline alert
            flagged, squad_is_live = await check_squad_availability(
                bootstrap_data, team_id, gw_info["gw"]
            )
            if not squad_is_live:
                print("[squad_check] Using public GW-1 fallback (may not reflect pending transfers)")

            success = await send_deadline_alert(
                webhook_url=webhook_url,
                gw_name=gw_info["name"],
                gw_number=gw_info["gw"],
                deadline_epoch=gw_info["deadline_epoch"],
                seconds_remaining=seconds_remaining,
                window_label=label,
            )
            if success:
                print(f"✓ Deadline notification sent: {gw_info['name']} in {label}")
            else:
                print("✗ Deadline notification failed.")
                sys.exit(1)

            # Send squad alert alongside deadline (if issues exist)
            if flagged:
                captain_affected = any(p.get("is_captain") for p in flagged)
                names = ", ".join(p["name"] for p in flagged)
                print(f"⚠️  {len(flagged)} player(s) flagged: {names}")
                squad_success = await send_squad_alert(
                    webhook_url=webhook_url,
                    gw_name=gw_info["name"],
                    flagged_players=flagged,
                    captain_affected=captain_affected,
                    squad_is_live=squad_is_live,
                )
                if squad_success:
                    print(f"✓ Squad alert sent ({len(flagged)} flagged)")
                else:
                    print("✗ Squad alert failed.")
            else:
                print("✓ All squad players available — no squad alert needed.")
            return

    print(f"No notification window matched ({hours_remaining:.1f}h remaining) — nothing sent.")


if __name__ == "__main__":
    asyncio.run(main())
