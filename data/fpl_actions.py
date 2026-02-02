"""
Authenticated FPL actions - transfers, captain, chips

Auth flow:
1. Try saved session from ~/.fpl_session (instant, no login needed)
2. If expired/missing, open a real Chrome window via Selenium for you to log in
3. Save session for next time — you only log in when session expires

FPL migrated to OAuth2 (account.premierleague.com) in 2025.
Auth tokens are stored in the browser's localStorage, not cookies.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

SESSION_FILE = Path.home() / ".fpl_session"
COOKIE_FILE = Path.home() / ".fpl_cookies"  # Legacy, checked for migration
FPL_DOMAIN = "fantasy.premierleague.com"


class FPLActions:
    """Execute actions on your FPL team (requires login)"""

    BASE_URL = f"https://{FPL_DOMAIN}/api"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36"
    )

    def __init__(self, team_id: int):
        self.team_id = team_id
        self._session: Optional[aiohttp.ClientSession] = None
        self._logged_in = False

    async def login(self) -> bool:
        """Authenticate with FPL via Selenium browser login"""
        # Always open a fresh browser login
        print("   Opening Chrome for FPL login...")
        session_data = _selenium_login()
        if not session_data:
            print("❌ Could not log in via browser")
            return False

        if await self._try_session(session_data):
            _save_session(session_data)
            print("✓ Logged in via Chrome (session saved)")
            return True

        print("❌ Browser session was not accepted by FPL API")
        return False

    async def _try_session(self, session_data: Dict[str, str]) -> bool:
        """Test if session data gives authenticated access.
        Tries cookies first (legacy), then Bearer token (OAuth2)."""
        if self._session:
            await self._session.close()

        headers = {
            "User-Agent": self.USER_AGENT,
            "Referer": f"https://{FPL_DOMAIN}/",
            "Origin": f"https://{FPL_DOMAIN}",
            "X-Requested-With": "XMLHttpRequest",
        }

        # Build cookie string from any cookies we have
        cookie_keys = [k for k in session_data if not k.startswith("_token")]
        if cookie_keys:
            cookie_str = "; ".join(f"{k}={session_data[k]}" for k in cookie_keys)
            headers["Cookie"] = cookie_str

        # Add CSRF header if available (legacy flow)
        csrf = session_data.get("csrftoken")
        if csrf:
            headers["X-CSRFToken"] = csrf

        # Add Bearer token if available (OAuth2 flow)
        access_token = session_data.get("_token_access")
        if access_token:
            headers["Authorization"] = f"Bearer {access_token}"

        jar = aiohttp.CookieJar(unsafe=True)
        self._session = aiohttp.ClientSession(cookie_jar=jar, headers=headers)

        # Verify with an authenticated endpoint
        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        try:
            async with self._session.get(url) as resp:
                if resp.status == 200:
                    self._logged_in = True
                    return True
        except Exception:
            pass

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

        # Use the authenticated /my-team/ endpoint to get the ACTUAL current squad
        # (the public /event/{gw}/picks/ endpoint shows the squad at GW deadline,
        # NOT the current squad after any pending transfers)
        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                print(f"❌ Could not fetch current picks (HTTP {resp.status})")
                return False
            data = await resp.json()

        actual_ids = [p["element"] for p in data.get("picks", [])]
        print(f"   [debug] set_captain: captain={captain_id} vc={vice_captain_id}")
        print(f"   [debug] set_captain: actual squad IDs={actual_ids}")

        # Validate captain/vc are in the actual squad
        if captain_id not in actual_ids:
            print(f"❌ Captain {captain_id} is not in the actual squad — skipping")
            return False
        if vice_captain_id not in actual_ids:
            print(f"   [debug] VC {vice_captain_id} not in squad, using captain as VC")
            vice_captain_id = captain_id

        # Check if captain is already set correctly
        already_correct = False
        for p in data["picks"]:
            if p["element"] == captain_id and p.get("is_captain"):
                already_correct = True
                break

        if already_correct:
            print("✓ Captain already set correctly")
            return True

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
        payload = {"picks": picks}

        async with self._session.post(url, json=payload) as resp:
            if resp.status in (200, 202):
                print("✓ Captain set successfully")
                return True
            else:
                text = await resp.text()
                print(f"❌ Failed to set captain (HTTP {resp.status}): {text[:200]}")
                return False

    async def make_transfers(
        self,
        transfers_out: List[int],
        transfers_in: List[int],
        current_gw: int,
        wildcard: bool = False,
        free_hit: bool = False,
    ) -> bool:
        """Execute transfers for the next gameweek"""
        self._check_login()

        if len(transfers_out) != len(transfers_in):
            print("❌ Transfers in/out must be equal length")
            return False

        # Get current squad info (selling prices) from authenticated endpoint
        url = f"{self.BASE_URL}/my-team/{self.team_id}/"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                print(f"❌ Could not fetch team data for transfers (HTTP {resp.status})")
                return False
            my_team = await resp.json()

        # Extract the ACTUAL current squad from the live API
        actual_squad_ids = [p["element"] for p in my_team.get("picks", [])]

        # Fetch current player prices + names from public API
        async with aiohttp.ClientSession() as pub:
            async with pub.get(f"https://{FPL_DOMAIN}/api/bootstrap-static/") as resp:
                bootstrap = await resp.json()
                price_map = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
                name_map = {e["id"]: e["web_name"] for e in bootstrap["elements"]}

        # Debug: show actual squad vs requested transfers
        print(f"   [debug] actual squad ({len(actual_squad_ids)} players): "
              f"{[f'{eid}({name_map.get(eid, '?')})' for eid in actual_squad_ids]}")
        print(f"   [debug] transfers_out: {[f'{eid}({name_map.get(eid, '?')})' for eid in transfers_out]}")
        print(f"   [debug] transfers_in:  {[f'{eid}({name_map.get(eid, '?')})' for eid in transfers_in]}")

        # Filter out transfers that have already been applied
        # (e.g. from a previous auto-pilot run with stale recommendations)
        valid_out = []
        valid_in = []
        for out_id, in_id in zip(transfers_out, transfers_in):
            out_in_squad = out_id in actual_squad_ids
            in_already = in_id in actual_squad_ids
            if out_in_squad and not in_already:
                valid_out.append(out_id)
                valid_in.append(in_id)
            elif not out_in_squad and in_already:
                print(f"   [debug] Skipping already-applied transfer: "
                      f"{name_map.get(out_id, out_id)} -> {name_map.get(in_id, in_id)}")
            else:
                print(f"   [debug] ⚠️  Unexpected state: OUT {out_id}({name_map.get(out_id, '?')}) "
                      f"in_squad={out_in_squad}, IN {in_id}({name_map.get(in_id, '?')}) "
                      f"in_squad={in_already}")

        if not valid_out:
            print("✓ All recommended transfers have already been applied — nothing to do")
            return True

        if len(valid_out) < len(transfers_out):
            print(f"   [debug] {len(transfers_out) - len(valid_out)} transfer(s) already applied, "
                  f"{len(valid_out)} remaining")

        transfers_out = valid_out
        transfers_in = valid_in

        # Build transfer list
        transfer_list = []
        for out_id, in_id in zip(transfers_out, transfers_in):
            selling_price = None
            for p in my_team.get("picks", []):
                if p["element"] == out_id:
                    selling_price = p["selling_price"]
                    break

            purchase_price = price_map.get(in_id, 0)

            transfer_list.append({
                "element_in": in_id,
                "element_out": out_id,
                "purchase_price": purchase_price,
                "selling_price": selling_price or 0,
            })

        chip = None
        if wildcard:
            chip = "wildcard"
        elif free_hit:
            chip = "freehit"

        # Target the next gameweek for transfers
        transfer_event = current_gw + 1
        transfers_info = my_team.get("transfers", {})

        print(f"   [debug] transfer_event=GW{transfer_event} | chip={chip} | num_transfers={len(transfer_list)}")
        print(f"   [debug] free_transfers={transfers_info.get('limit', '?')} | bank=£{transfers_info.get('bank', 0) / 10:.1f}m")

        payload = {
            "confirmed": True,
            "entry": self.team_id,
            "event": transfer_event,
            "transfers": transfer_list,
        }
        if chip:
            payload["chip"] = chip

        url = f"{self.BASE_URL}/transfers/"
        async with self._session.post(url, json=payload) as resp:
            if resp.status in (200, 202):
                hits = max(0, (len(transfers_out) - transfers_info.get("limit", 1)) * 4)
                print(f"✓ {len(transfers_out)} transfer(s) made" +
                      (f" (-{hits} pts)" if hits > 0 else " (free)"))
                return True
            else:
                text = await resp.text()
                print(f"❌ Transfer failed (HTTP {resp.status}): {text[:200]}")
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


# --- Session persistence ---

def _save_session(session_data: Dict[str, str]):
    """Save session data to ~/.fpl_session"""
    data = {"session": session_data, "saved_at": time.time()}
    SESSION_FILE.write_text(json.dumps(data))
    os.chmod(SESSION_FILE, 0o600)


def _load_session() -> Optional[Dict[str, str]]:
    """Load session from ~/.fpl_session if it exists and isn't too old"""
    # Try new session file first
    if SESSION_FILE.exists():
        try:
            data = json.loads(SESSION_FILE.read_text())
            if time.time() - data.get("saved_at", 0) > 7 * 86400:
                return None
            return data.get("session")
        except (json.JSONDecodeError, KeyError):
            pass

    # Fall back to legacy cookie file
    if COOKIE_FILE.exists():
        try:
            data = json.loads(COOKIE_FILE.read_text())
            if time.time() - data.get("saved_at", 0) > 7 * 86400:
                return None
            return data.get("cookies")
        except (json.JSONDecodeError, KeyError):
            pass

    return None


# --- Selenium browser login ---

def _selenium_login() -> Optional[Dict[str, str]]:
    """Open Chrome, let user log in, capture cookies + localStorage tokens"""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except ImportError:
        print("   selenium not installed (pip install selenium)")
        return None

    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"   Could not start Chrome: {e}")
        print("   Make sure Chrome is installed")
        return None

    try:
        # Navigate to FPL — it will redirect to the OAuth login page
        FPL_TARGET = "https://fantasy.premierleague.com/my-team"
        driver.get(FPL_TARGET)

        print("   Log in to FPL in the Chrome window...")
        print("   (the window will close automatically once you're logged in)")

        deadline = time.time() + 180
        fpl_landed_at = 0

        while time.time() < deadline:
            time.sleep(2)

            try:
                current = driver.current_url
            except Exception:
                break

            if not current:
                continue

            if FPL_DOMAIN not in current:
                fpl_landed_at = 0
                continue

            # First time on FPL domain after login
            if fpl_landed_at == 0:
                fpl_landed_at = time.time()

            # Give JS time to exchange OAuth code for tokens
            if time.time() - fpl_landed_at < 5:
                continue

            # Extract cookies
            cookies = {}
            try:
                for c in driver.get_cookies():
                    cookies[c["name"]] = c["value"]
            except Exception:
                break

            # Check for legacy csrftoken first
            if "csrftoken" in cookies:
                return cookies

            # Extract OAuth tokens from localStorage
            try:
                local_storage = driver.execute_script(
                    "var items = {};"
                    "for (var i = 0; i < localStorage.length; i++) {"
                    "  var k = localStorage.key(i);"
                    "  items[k] = localStorage.getItem(k);"
                    "}"
                    "return items;"
                )
            except Exception:
                local_storage = {}

            # Look for access tokens in localStorage
            # OAuth/OIDC libraries store tokens under various key patterns
            access_token = None
            id_token = None
            for key, value in local_storage.items():
                if not value:
                    continue
                # Try parsing JSON values (OIDC client libraries store JSON)
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        if "access_token" in parsed:
                            access_token = parsed["access_token"]
                        if "id_token" in parsed:
                            id_token = parsed["id_token"]
                except (json.JSONDecodeError, TypeError):
                    pass
                # Direct token keys
                key_lower = key.lower()
                if "access_token" in key_lower and not access_token:
                    access_token = value
                elif "id_token" in key_lower and not id_token:
                    id_token = value

            if access_token:
                # Merge cookies + tokens into session data
                session_data = dict(cookies)
                session_data["_token_access"] = access_token
                if id_token:
                    session_data["_token_id"] = id_token
                return session_data

            # After 20s on FPL, also try navigating to /my-team
            # to trigger any lazy session setup
            if time.time() - fpl_landed_at > 20 and current != FPL_TARGET:
                try:
                    driver.get(FPL_TARGET)
                except Exception:
                    break

            # After 45s, return whatever we have
            if time.time() - fpl_landed_at > 45 and cookies:
                return cookies

        print("   Timed out waiting for login")
        return None

    finally:
        try:
            driver.quit()
        except Exception:
            pass
