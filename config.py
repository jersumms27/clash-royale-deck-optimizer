"""Tunable settings for the deck optimizer: paths, deck rules, GA params."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
CARDS_CSV = ROOT / "cards.csv"

# Clash Royale API (tokens are IP-locked; get one at developer.clashroyale.com)
CR_API_BASE = os.environ.get("CR_API_BASE", "https://api.clashroyale.com/v1")

# Where to look for the API token (never commit the token itself).
TOKEN_FILE = ROOT / "token.txt"


def get_api_token() -> str | None:
    """Return the Clash Royale API token, or None if it isn't configured.

    Resolution order (first hit wins):
      1. the CR_API_TOKEN environment variable
      2. a token.txt file next to this script (gitignored)
    """
    env = os.environ.get("CR_API_TOKEN")
    if env and env.strip():
        return env.strip()

    try:
        text = TOKEN_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


# Deck rules: 1 evo slot + 1 champion slot + 1 wild slot (wild = 2nd evo OR 2nd champion)
DECK_SIZE = 8
BASE_EVOLUTION_SLOTS = 1
BASE_CHAMPION_SLOTS = 1
WILD_SLOTS = 1
MAX_EVOLUTIONS = BASE_EVOLUTION_SLOTS + WILD_SLOTS
MAX_CHAMPIONS = BASE_CHAMPION_SLOTS + WILD_SLOTS
CHAMPION_RARITY = "champion"

# Card type is encoded in the API id range (id // 1_000_000): troop/building/spell.
TYPE_BY_ID_PREFIX = {26: "troop", 27: "building", 28: "spell"}
# Exceptions: troops that live in the spell id-range (28xxxxxx) and so would
# otherwise be misclassified. Heal Spirit was the reworked "Heal" spell.
CARD_TYPE_OVERRIDES = {"Heal Spirit": "troop", "Spirit Empress": "troop"}

# Cards eligible for the champion/hero slot (name-based; not derivable from the API).
# Names must match the API's spelling exactly (e.g. "Mini P.E.K.K.A", no trailing dot).
CHAMPION_HERO_NAMES = {
    "Archer Queen", "Mighty Miner", "Skeleton King", "Golden Knight", "Monk",
    "Little Prince", "Goblinstein", "Boss Bandit", "Knight", "Giant",
    "Mini P.E.K.K.A", "Musketeer", "Ice Golem", "Wizard", "Goblins",
    "Mega Minion", "Barbarian Barrel", "Magic Archer", "Balloon", "Bowler",
    "Dark Prince", "Tombstone",
}

# Win-condition classification (name-based). Cards not listed get "" (empty).
_WIN_PRIMARY = {
    "Mortar", "Royal Giant", "Elixir Golem", "Battle Ram", "Hog Rider", "Giant",
    "Royal Hogs", "Wall Breakers", "Goblin Barrel", "Goblin Drill", "Balloon",
    "Goblin Giant", "X-Bow", "Electro Giant", "Golem", "Miner", "Ram Rider",
    "Graveyard", "Lava Hound",
}
_WIN_SECONDARY = {
    "Skeleton Barrel", "Suspicious Bush", "Goblin Demolisher", "Rocket",
    "Three Musketeers", "Rune Giant", "Prince", "Princess", "Bandit",
    "Magic Archer", "Goblinstein",
}
_WIN_CONDITIONAL = {
    "Firecracker", "Elite Barbarians", "Royal Recruits", "Ice Golem", "Dart Goblin",
    "Earthquake", "Dark Prince", "Lightning", "P.E.K.K.A", "Sparky", "Mega Knight",
    "Golden Knight", "Monk", "Boss Bandit",
}
WIN_CONDITION_BY_NAME = {
    **{n: "primary" for n in _WIN_PRIMARY},
    **{n: "secondary" for n in _WIN_SECONDARY},
    **{n: "conditional" for n in _WIN_CONDITIONAL},
}

# Air (flying) troops. Name-based; not derivable from API/scrape data.
AIR_UNIT_NAMES = {
    "Minions", "Minion Horde", "Mega Minion", "Bats", "Baby Dragon",
    "Inferno Dragon", "Electro Dragon", "Skeleton Dragons", "Balloon",
    "Lava Hound", "Flying Machine", "Phoenix", "Skeleton Barrel",
}


def slots_ok(num_evolutions: int, num_champions: int) -> bool:
    """True if the (#evolutions, #champions) combo fits the slot model."""
    extra_evo = max(0, num_evolutions - BASE_EVOLUTION_SLOTS)
    extra_champ = max(0, num_champions - BASE_CHAMPION_SLOTS)
    return (
        num_evolutions <= MAX_EVOLUTIONS
        and num_champions <= MAX_CHAMPIONS
        and extra_evo + extra_champ <= WILD_SLOTS
    )


def max_evolutions_allowed(num_champions: int) -> int:
    """Evolutions allowed given how many champions are in the deck."""
    wild_used_by_champion = max(0, num_champions - BASE_CHAMPION_SLOTS)
    return BASE_EVOLUTION_SLOTS + (WILD_SLOTS - wild_used_by_champion)


# Genetic algorithm
POPULATION_SIZE: int = 1000
GENERATIONS: int = 1000
ELITISM: int = 4
TOURNAMENT_SIZE: int = 5
CROSSOVER_RATE: int = 0.85
MUTATION_RATE: int = 0.50
RANDOM_SEED: int | None = None
