#!/usr/bin/env python3
"""
FPL Deadline Checker — sends Discord notifications when a GW deadline approaches,
and squad availability alerts when players are flagged.

Notification windows (triggered once per 6-hour cron cycle):
  - 48h warning  (42h < remaining ≤ 48h)
  - 24h warning  (18h < remaining ≤ 24h)
  -  6h warning  ( 0h < remaining ≤  6h)

Squad alerts fire on EVERY cron run where issues are detected,
regardless of notification windows.

Usage:
  DISCORD_WEBHOOK_URL=<url> python3 check_deadline.py
  python3 check_deadline.py --test   # send test messages immediately
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

import aiohttp

import config
from data.discord_notifier import send_deadline_alert, send_squad_alert

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FPL_PICKS_URL = "https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"

# (upper_seconds, lower_seconds, label)
NOTIFICATION_WINDOWS = [
    (48 * 3600, 42 * 3600, "48 hours"),
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


async def check_squad_availability(
    bootstrap_data: dict,
    team_id: int,
    current_gw: int,
) -> list[dict]:
    """
    Check squad players for availability issues.

    Returns list of dicts: {name, status, chance, news, is_captain, is_vice_captain, position}
    """
    # Build element lookup
    elements = {el["id"]: el for el in bootstrap_data["elements"]}

    # Fetch squad picks — try previous GW (last confirmed lineup).
    # current_gw picks don't exist until after the deadline, so always
    # use current_gw - 1 as the base squad.  Any pending transfers for
    # current_gw are applied below via the transfers endpoint.
    picks_data = None
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

    # Apply any pending transfers for current_gw so the squad is up to date
    if picks_data:
        squad_element_ids = {p["element"] for p in picks_data["picks"]}
        try:
            transfers_url = f"https://fantasy.premierleague.com/api/entry/{team_id}/transfers/"
            async with aiohttp.ClientSession() as session:
                async with session.get(transfers_url) as resp:
                    if resp.status == 200:
                        transfers = await resp.json(content_type=None)
                        for t in transfers:
                            if t.get("event") == current_gw:
                                squad_element_ids.discard(t["element_out"])
                                squad_element_ids.add(t["element_in"])
        except Exception:
            pass
        # Rebuild picks list with updated squad (drop stale captain/VC flags —
        # they're from last GW and meaningless before the new deadline)
        picks_data["picks"] = [
            {"element": eid, "is_captain": False, "is_vice_captain": False}
            for eid in squad_element_ids
        ]

    if not picks_data:
        print(f"[squad_check] Could not fetch picks for team {team_id}")
        return []

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

    return flagged


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

    # --- Squad availability check (runs every cron cycle) ---
    if gw_info:
        print(f"Checking squad availability for GW{gw_info['gw']}...")
        flagged = await check_squad_availability(
            bootstrap_data, team_id, gw_info["gw"]
        )

        if flagged:
            captain_affected = any(p.get("is_captain") for p in flagged)
            names = ", ".join(p["name"] for p in flagged)
            print(f"⚠️  {len(flagged)} player(s) flagged: {names}")
            if captain_affected:
                print("🔴 CAPTAIN is affected!")

            success = await send_squad_alert(
                webhook_url=webhook_url,
                gw_name=gw_info["name"],
                flagged_players=flagged,
                captain_affected=captain_affected,
            )
            if success:
                print(f"✓ Squad alert sent ({len(flagged)} flagged)")
            else:
                print("✗ Squad alert failed.")
        else:
            print("✓ All squad players available.")

    # --- Deadline notification (only in notification windows) ---

    # --now flag: send the real current deadline immediately, ignoring windows
    if "--now" in sys.argv:
        if not gw_info:
            print("No upcoming gameweek found.")
            return
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        seconds_remaining = gw_info["deadline_epoch"] - now_epoch
        hours_remaining = seconds_remaining / 3600
        label = f"{hours_remaining:.0f} hours"
        success = await send_deadline_alert(
            webhook_url=webhook_url,
            gw_name=gw_info["name"],
            gw_number=gw_info["gw"],
            deadline_epoch=gw_info["deadline_epoch"],
            seconds_remaining=seconds_remaining,
            window_label=label,
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

    for upper, lower, label in NOTIFICATION_WINDOWS:
        if lower < seconds_remaining <= upper:
            print(f"→ In '{label}' window — sending Discord notification...")
            success = await send_deadline_alert(
                webhook_url=webhook_url,
                gw_name=gw_info["name"],
                gw_number=gw_info["gw"],
                deadline_epoch=gw_info["deadline_epoch"],
                seconds_remaining=seconds_remaining,
                window_label=label,
            )
            if success:
                print(f"✓ Notification sent: {gw_info['name']} in {label}")
            else:
                print("✗ Notification failed.")
                sys.exit(1)
            return

    print(f"No notification window matched ({hours_remaining:.1f}h remaining) — nothing sent.")


if __name__ == "__main__":
    asyncio.run(main())
