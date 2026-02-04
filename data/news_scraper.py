"""
News scraper for injury updates and team news
"""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup

import config


@dataclass
class NewsItem:
    """News article container"""

    title: str
    link: str
    published: Optional[datetime]
    summary: str


@dataclass
class InjuryStatus:
    """Player injury information"""

    player_name: str
    team: str
    status: str  # "fit", "doubt", "injured", "suspended"
    details: str
    return_date: Optional[str]
    chance_of_playing: Optional[int]


class NewsScraper:
    """Scrapes FPL news from Fantasy Football Scout"""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    def __init__(self):
        self.rss_url = config.NEWS_RSS_URL
        self.injuries_url = config.INJURIES_URL
        self.team_news_url = config.TEAM_NEWS_URL

    def fetch_rss_feed(self, limit: int = 10) -> List[NewsItem]:
        """Fetch latest news from RSS feed"""
        try:
            feed = feedparser.parse(self.rss_url)
            news_items = []

            for entry in feed.entries[:limit]:
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])

                news_items.append(
                    NewsItem(
                        title=entry.title,
                        link=entry.link,
                        published=published,
                        summary=entry.get("summary", "")[:200],
                    )
                )

            return news_items

        except Exception as e:
            print(f"Error fetching RSS feed: {e}")
            return []

    def scrape_injuries(self) -> Dict[str, InjuryStatus]:
        """Scrape injury information from FFS"""
        injuries: Dict[str, InjuryStatus] = {}

        try:
            response = requests.get(self.injuries_url, headers=self.HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            # Look for injury tables/lists
            # FFS typically uses tables for injury data
            tables = soup.find_all("table")

            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(["td", "th"])
                    if len(cells) >= 3:
                        player_name = cells[0].get_text(strip=True)
                        team = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                        status_text = (
                            cells[2].get_text(strip=True) if len(cells) > 2 else ""
                        )
                        details = (
                            cells[3].get_text(strip=True) if len(cells) > 3 else ""
                        )
                        return_date = (
                            cells[4].get_text(strip=True) if len(cells) > 4 else None
                        )

                        # Determine status category
                        status = self._categorize_status(status_text)

                        injuries[player_name.lower()] = InjuryStatus(
                            player_name=player_name,
                            team=team,
                            status=status,
                            details=details or status_text,
                            return_date=return_date,
                            chance_of_playing=self._parse_chance(status_text),
                        )

            # Also look for div-based layouts
            injury_items = soup.find_all(class_=re.compile(r"injury|player-status"))
            for item in injury_items:
                name_elem = item.find(class_=re.compile(r"name|player"))
                status_elem = item.find(class_=re.compile(r"status|injury"))

                if name_elem:
                    player_name = name_elem.get_text(strip=True)
                    status_text = (
                        status_elem.get_text(strip=True) if status_elem else "unknown"
                    )
                    status = self._categorize_status(status_text)

                    injuries[player_name.lower()] = InjuryStatus(
                        player_name=player_name,
                        team="",
                        status=status,
                        details=status_text,
                        return_date=None,
                        chance_of_playing=self._parse_chance(status_text),
                    )

        except Exception as e:
            print(f"Error scraping injuries: {e}")

        return injuries

    def _categorize_status(self, status_text: str) -> str:
        """Categorize status text into standard categories"""
        status_lower = status_text.lower()

        if any(word in status_lower for word in ["suspended", "ban", "red card"]):
            return "suspended"
        elif any(
            word in status_lower
            for word in ["out", "injured", "injury", "ruled out", "sidelined"]
        ):
            return "injured"
        elif any(word in status_lower for word in ["doubt", "50%", "75%", "knock"]):
            return "doubt"
        else:
            return "fit"

    def _parse_chance(self, status_text: str) -> Optional[int]:
        """Extract chance of playing percentage from text"""
        match = re.search(r"(\d+)%", status_text)
        if match:
            return int(match.group(1))
        return None

    def get_availability_flag(self, player_name: str) -> str:
        """Get availability status for a specific player"""
        injuries = self.scrape_injuries()
        player_key = player_name.lower()

        if player_key in injuries:
            return injuries[player_key].status

        # Check partial matches
        for key, injury in injuries.items():
            if player_key in key or key in player_key:
                return injury.status

        return "fit"

    def get_flagged_players(self, player_names: List[str]) -> Dict[str, InjuryStatus]:
        """Check availability for a list of players"""
        injuries = self.scrape_injuries()
        flagged = {}

        for name in player_names:
            name_lower = name.lower()
            if name_lower in injuries:
                flagged[name] = injuries[name_lower]
            else:
                # Check partial matches
                for key, injury in injuries.items():
                    if name_lower in key or key in name_lower:
                        flagged[name] = injury
                        break

        return flagged

    def get_injury_summary(self) -> str:
        """Get formatted summary of current injuries"""
        injuries = self.scrape_injuries()

        if not injuries:
            return "No injury data available"

        lines = ["Current Injuries/Doubts:"]

        # Group by status
        by_status: Dict[str, List[InjuryStatus]] = {
            "injured": [],
            "suspended": [],
            "doubt": [],
        }

        for injury in injuries.values():
            if injury.status in by_status:
                by_status[injury.status].append(injury)

        for status, players in by_status.items():
            if players:
                lines.append(f"\n{status.upper()}:")
                for p in players[:10]:  # Limit output
                    lines.append(f"  - {p.player_name} ({p.team}): {p.details}")

        return "\n".join(lines)


def get_news_scraper() -> NewsScraper:
    """Get a NewsScraper instance"""
    return NewsScraper()
