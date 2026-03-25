#!/usr/bin/env python3
"""
FPL Comeback Assistant — Discord Bot

Commands (type in Discord):
  deadline  — send the current GW deadline notification
  squad     — check squad availability right now
  analysis  — run full FPL analysis and post output
  help      — show available commands
"""

import asyncio
import os
import sys
from pathlib import Path

import discord
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
CHANNEL_ID = int(os.environ["DISCORD_CHANNEL_ID"])
SCRIPT_DIR = Path(__file__).parent
VENV_PYTHON = SCRIPT_DIR / "venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

HELP_TEXT = """**FPL Bot Commands**
`deadline` — send the current GW deadline notification
`squad` — check squad availability right now
`analysis` — run full FPL analysis (captain picks, transfers, etc.)
`help` — show this message"""

DISCORD_MAX_LEN = 1900


def chunk_output(text: str) -> list:
    lines = text.splitlines(keepends=True)
    chunks, current = [], ""
    for line in lines:
        if len(current) + len(line) > DISCORD_MAX_LEN:
            if current:
                chunks.append(current.rstrip())
            current = line
        else:
            current += line
    if current.strip():
        chunks.append(current.rstrip())
    return chunks or ["(no output)"]


async def run_script(args, channel):
    cmd = [PYTHON] + args
    await channel.send(f"⚙️ Running `{Path(args[0]).name}`...")
    env = os.environ.copy()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(SCRIPT_DIR),
        env=env,
    )
    stdout, _ = await proc.communicate()
    output = stdout.decode(errors="replace").strip() or "(no output)"
    for chunk in chunk_output(output):
        await channel.send(f"```\n{chunk}\n```")
    await channel.send("✅ Done" if proc.returncode == 0 else f"⚠️ Exited with code {proc.returncode}")


class FPLBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self._announced = False  # Only announce once per process lifetime

    async def on_ready(self):
        print(f"[fpl-bot] Logged in as {self.user}")
        if self._announced:
            print("[fpl-bot] Reconnected (suppressing duplicate online message)")
            return
        self._announced = True

    async def on_message(self, message):
        if message.author == self.user:
            return
        if message.channel.id != CHANNEL_ID:
            return

        text = message.content.strip().lower()

        if text == "help":
            await message.channel.send(HELP_TEXT)

        elif text == "deadline":
            await message.channel.send("📅 Fetching deadline info...")
            await run_script(["check_deadline.py", "--now"], message.channel)

        elif text == "squad":
            await message.channel.send("🔍 Checking squad availability...")
            await run_script(["check_deadline.py"], message.channel)

        elif text == "analysis":
            await message.channel.send("📊 Running full analysis — give me ~60 seconds...")
            await run_script(["main.py", "--team-id", "7907269"], message.channel)


if __name__ == "__main__":
    missing = [v for v in ["DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID"] if not os.environ.get(v)]
    if missing:
        print(f"Error: missing env vars: {', '.join(missing)}")
        sys.exit(1)
    FPLBot().run(BOT_TOKEN)
