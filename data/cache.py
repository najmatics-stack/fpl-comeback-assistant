"""
Local caching to avoid rate limits and improve performance
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import config


class Cache:
    """Simple file-based cache for API responses"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.expiry_hours = config.CACHE_EXPIRY_HOURS

    def _get_cache_path(self, key: str) -> Path:
        """Get the file path for a cache key"""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"

    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from cache if valid"""
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - cached_time > timedelta(hours=self.expiry_hours):
                # Cache expired
                cache_path.unlink()
                return None

            return cached["data"]

        except (json.JSONDecodeError, KeyError, ValueError):
            # Invalid cache file
            cache_path.unlink()
            return None

    def set(self, key: str, data: Any) -> None:
        """Store data in cache"""
        cache_path = self._get_cache_path(key)

        cached = {"timestamp": datetime.now().isoformat(), "data": data}

        with open(cache_path, "w") as f:
            json.dump(cached, f)

    def clear(self) -> None:
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def clear_expired(self) -> None:
        """Remove only expired cache entries"""
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                cached_time = datetime.fromisoformat(cached["timestamp"])
                if datetime.now() - cached_time > timedelta(hours=self.expiry_hours):
                    cache_file.unlink()
            except (json.JSONDecodeError, KeyError, ValueError):
                cache_file.unlink()
