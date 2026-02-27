#!/usr/bin/env python3
"""
FPL Deadline Checker — sends Discord notifications when a GW deadline approaches.

Notification windows (triggered once per 6-hour cron cycle):
  - 48h warning  (42h < remaining ≤ 48h)
  - 24h warning  (18h < remaining ≤ 24h)
  -  6h warning  ( 0h < remaining ≤  6h)

Usage:
  DISCORD_WEBHOOK_URL=<url> python3 check_deadline.py
  python3 check_deadline.py --test   # send a test message immediately
"""

import asyncio
import os
import sys
from datetime import datetime, timezone

import aiohttp

from data.discord_notifier import send_deadline_alert

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

# (upper_seconds, lower_seconds, label)
NOTIFICATION_WINDOWS = [
    (48 * 3600, 42 * 3600, "48 hours"),
    (24 * 3600, 18 * 3600, "24 hours"),
    (6 * 3600, 0, "6 hours"),
]


async def fetch_next_deadline() -> dict | None:
    """Fetch the next upcoming gameweek deadline from the FPL API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(FPL_BOOTSTRAP_URL) as resp:
            data = await resp.json(content_type=None)

    for event in data["events"]:
        if event.get("is_next"):
            return {
                "gw": event["id"],
                "name": event["name"],
                "deadline_epoch": event["deadline_time_epoch"],
                "deadline_str": event["deadline_time"],
            }
    return None


async def main() -> None:
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("Error: DISCORD_WEBHOOK_URL environment variable not set.")
        sys.exit(1)

    # --now flag: send the real current deadline immediately, ignoring windows
    if "--now" in sys.argv:
        print("Fetching FPL bootstrap data...")
        gw_info = await fetch_next_deadline()
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

    # --test flag: send a test message and exit
    if "--test" in sys.argv:
        now = int(datetime.now(timezone.utc).timestamp())
        success = await send_deadline_alert(
            webhook_url=webhook_url,
            gw_name="Gameweek 99 (TEST)",
            gw_number=99,
            deadline_epoch=now + 6 * 3600,
            seconds_remaining=6 * 3600,
            window_label="6 hours",
        )
        print("✓ Test message sent." if success else "✗ Test message failed.")
        return

    print("Fetching FPL bootstrap data...")
    gw_info = await fetch_next_deadline()

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
