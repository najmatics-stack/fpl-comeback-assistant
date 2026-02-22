"""
Discord webhook notifications for FPL deadline alerts.
Uses stdlib only — no extra dependencies.
"""

import json
import urllib.request
from datetime import datetime, timezone


def send_deadline_alert(
    webhook_url: str,
    gw_name: str,
    gw_number: int,
    deadline_epoch: int,
    seconds_remaining: int,
    window_label: str,
) -> bool:
    """
    Send a formatted Discord embed for an upcoming FPL deadline.
    Returns True on success, False on failure.
    """
    hours = seconds_remaining // 3600
    minutes = (seconds_remaining % 3600) // 60

    deadline_dt = datetime.fromtimestamp(deadline_epoch, tz=timezone.utc)
    deadline_formatted = deadline_dt.strftime("%a %d %b at %H:%M UTC")

    # Color: red < 6h, orange < 24h, yellow < 48h
    if seconds_remaining <= 6 * 3600:
        color = 0xFF0000  # red
    elif seconds_remaining <= 24 * 3600:
        color = 0xFF8C00  # orange
    else:
        color = 0xFFCC00  # yellow

    payload = {
        "embeds": [
            {
                "title": f"🚨 FPL Deadline in {window_label}!",
                "description": (
                    f"**{gw_name}** deadline is approaching — don't get caught out."
                ),
                "color": color,
                "fields": [
                    {
                        "name": "📅 Deadline",
                        "value": deadline_formatted,
                        "inline": True,
                    },
                    {
                        "name": "⏳ Time Left",
                        "value": f"{hours}h {minutes}m",
                        "inline": True,
                    },
                    {
                        "name": "🤖 Quick Action",
                        "value": "`python3 main.py --auto`",
                        "inline": False,
                    },
                ],
                "footer": {"text": "FC ZBEEB | FPL Comeback Assistant"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 204)
    except Exception as e:
        print(f"[discord_notifier] Failed to send: {e}")
        return False
