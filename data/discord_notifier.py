"""
Discord webhook notifications for FPL deadline alerts.
"""

import json
from datetime import datetime, timezone

import aiohttp


async def send_deadline_alert(
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
        "content": "@everyone",
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
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status in (200, 204)
    except Exception as e:
        print(f"[discord_notifier] Failed to send: {e}")
        return False


STATUS_LABELS = {
    "i": "Injured",
    "s": "Suspended",
    "d": "Doubtful",
    "u": "Unavailable",
}


async def send_squad_alert(
    webhook_url: str,
    gw_name: str,
    flagged_players: list[dict],
    captain_affected: bool,
) -> bool:
    """
    Send a Discord embed alerting about unavailable squad players.

    flagged_players: list of dicts with keys:
        name, status, chance, news, is_captain, is_vice_captain, position

    Returns True on success, False on failure.
    """
    if not flagged_players:
        return True

    color = 0xFF0000 if captain_affected else 0xFF8C00  # red / orange

    fields = []

    # Captain warning first
    for p in flagged_players:
        if p.get("is_captain"):
            status_text = STATUS_LABELS.get(p["status"], p["status"])
            chance_str = f"{p['chance']}%" if p["chance"] is not None else "unknown"
            fields.append({
                "name": f"\U0001f534 CAPTAIN {p['name']} — {status_text} ({chance_str} chance)",
                "value": p.get("news") or "No further details",
                "inline": False,
            })

    # All flagged players
    for p in flagged_players:
        if p.get("is_captain"):
            continue  # already shown above
        status_text = STATUS_LABELS.get(p["status"], p["status"])
        chance_str = f"{p['chance']}%" if p["chance"] is not None else "unknown"
        role = ""
        if p.get("is_vice_captain"):
            role = " (VC)"
        fields.append({
            "name": f"{p['position']} — {p['name']}{role}",
            "value": f"{status_text} ({chance_str} chance) — {p.get('news') or 'No details'}",
            "inline": False,
        })

    title = (
        f"\U0001f6a8 {gw_name}: Captain unavailable!"
        if captain_affected
        else f"\u26a0\ufe0f {gw_name}: Squad availability issues"
    )

    payload = {
        "content": "@everyone",
        "embeds": [
            {
                "title": title,
                "description": f"**{len(flagged_players)}** player(s) flagged in your squad.",
                "color": color,
                "fields": fields[:10],  # Discord max 25, keep reasonable
                "footer": {"text": "FC ZBEEB | FPL Comeback Assistant"},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ],
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                webhook_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status in (200, 204)
    except Exception as e:
        print(f"[discord_notifier] Failed to send squad alert: {e}")
        return False
