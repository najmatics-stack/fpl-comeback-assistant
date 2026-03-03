"""Data fetching modules"""

# Lazy imports — consumers use explicit submodule imports (e.g. from data.fpl_api import ...).
# Eager re-exports here would pull in heavy deps (pandas) and break lightweight
# scripts like check_deadline.py that only need data.discord_notifier + aiohttp.
