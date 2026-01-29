"""
Authenticated FPL actions - transfers, captain, chips
"""

import asyncio
import getpass
from typing import List, Optional, Tuple

import aiohttp


class FPLActions:
    """Execute actions on your FPL team (requires login)"""

    BASE_URL = "https://fantasy.premierleague.com/api"
    LOGIN_URL = "https://users.premierleague.com/accounts/login/"

    def __init__(self, team_id: int):
        self.team_id = team_id
        self._session: Optional[aiohttp.ClientSession] = None
        self._logged_in = False

    async def login(self, email: str, password: str) -> bool:
        """Authenticate with FPL"""
        self._session = aiohttp.ClientSession()

        payload = {
            "login": email,
            "password": password,
            "redirect_uri": "https://fantasy.premierleague.com/",
            "app": "plfpl-web",
        }

        async with self._session.post(
            self.LOGIN_URL,
            data=payload,
            allow_redirects=False,
        ) as resp:
            if resp.status in (200, 302):
                self._logged_in = True
                print("✓ Logged in to FPL")
                return True
            else:
                print(f"❌ Login failed (status {resp.status})")
                return False

    async def close(self):
        """Close the session"""
        if self._session:
            await self._session.close()

    def _check_login(self):
        if not self._logged_in or not self._session:
            raise RuntimeError("Not logged in. Call login() first.")

    async def set_captain(self, captain_id: int, vice_captain_id: int, current_gw: int) -> bool:
        """Set captain and vice captain"""
        self._check_login()

        # First get current picks to preserve them
        url = f"{self.BASE_URL}/entry/{self.team_id}/event/{current_gw}/picks/"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                print(f"❌ Could not fetch current picks")
                return False
            data = await resp.json()

        # Build updated picks
        picks = []
        for p in data["picks"]:
            pick = {
                "element": p["element"],
                "position": p["position"],
                "is_captain": p["element"] == captain_id,
                "is_vice_captain": p["element"] == vice_captain_id,
            }
            picks.append(pick)

        # Submit
        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        payload = {"picks": picks}

        async with self._session.post(url, json=payload) as resp:
            if resp.status == 200:
                print(f"✓ Captain set successfully")
                return True
            else:
                text = await resp.text()
                print(f"❌ Failed to set captain: {text[:100]}")
                return False

    async def make_transfers(
        self,
        transfers_out: List[int],
        transfers_in: List[int],
        wildcard: bool = False,
        free_hit: bool = False,
    ) -> bool:
        """Execute transfers"""
        self._check_login()

        if len(transfers_out) != len(transfers_in):
            print("❌ Transfers in/out must be equal length")
            return False

        # Get current squad value info
        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                print("❌ Could not fetch team data for transfers")
                return False
            my_team = await resp.json()

        # Build transfer list
        transfer_list = []
        for out_id, in_id in zip(transfers_out, transfers_in):
            # Find selling price from my team data
            selling_price = None
            for p in my_team.get("picks", []):
                if p["element"] == out_id:
                    selling_price = p["selling_price"]
                    break

            transfer_list.append({
                "element_in": in_id,
                "element_out": out_id,
                "purchase_price": selling_price or 0,
                "selling_price": selling_price or 0,
            })

        chip = None
        if wildcard:
            chip = "wildcard"
        elif free_hit:
            chip = "freehit"

        payload = {
            "confirmed": True,
            "entry": self.team_id,
            "event": my_team.get("current_event", 0) + 1,
            "transfers": transfer_list,
        }
        if chip:
            payload["chip"] = chip

        url = f"{self.BASE_URL}/transfers/"
        async with self._session.post(url, json=payload) as resp:
            if resp.status == 200:
                hits = max(0, (len(transfers_out) - my_team.get("transfers", {}).get("limit", 1)) * 4)
                print(f"✓ {len(transfers_out)} transfer(s) made" +
                      (f" (-{hits} pts)" if hits > 0 else " (free)"))
                return True
            else:
                text = await resp.text()
                print(f"❌ Transfer failed: {text[:150]}")
                return False

    async def activate_chip(self, chip_name: str, current_gw: int) -> bool:
        """Activate a chip (triple_captain, bench_boost)"""
        self._check_login()

        chip_map = {
            "triple_captain": "3xc",
            "bench_boost": "bboost",
            "free_hit": "freehit",
            "wildcard": "wildcard",
        }

        chip_code = chip_map.get(chip_name)
        if not chip_code:
            print(f"❌ Unknown chip: {chip_name}")
            return False

        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        payload = {"chip": chip_code}

        async with self._session.post(url, json=payload) as resp:
            if resp.status == 200:
                print(f"✓ {chip_name.replace('_', ' ').title()} activated")
                return True
            else:
                text = await resp.text()
                print(f"❌ Failed to activate chip: {text[:100]}")
                return False

    async def get_remaining_budget(self) -> float:
        """Get remaining transfer budget"""
        self._check_login()

        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        async with self._session.get(url) as resp:
            data = await resp.json()
            bank = data.get("transfers", {}).get("bank", 0)
            return bank / 10

    async def get_free_transfers(self) -> int:
        """Get number of free transfers available"""
        self._check_login()

        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        async with self._session.get(url) as resp:
            data = await resp.json()
            return data.get("transfers", {}).get("limit", 1)


def prompt_credentials() -> Tuple[str, str]:
    """Prompt user for FPL login credentials"""
    print("\n🔐 FPL Login Required")
    email = input("Email: ")
    password = getpass.getpass("Password: ")
    return email, password
