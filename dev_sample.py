"""Write an offline sample cards.csv for development before you have a token.
Run: python dev_sample.py  (later, `python main.py --refresh` replaces it with
the real list). Flags are approximate and may lag the live meta.
"""

from __future__ import annotations

from cr_api import save_cards_csv
from models import Card, build_card

# (name, elixir, rarity, has_evolution)
_SAMPLE: list[tuple[str, int, str, bool]] = [
    # --- commons ---
    ("Knight", 3, "common", True),
    ("Archers", 3, "common", True),
    ("Skeletons", 1, "common", True),
    ("Bomber", 2, "common", True),
    ("Barbarians", 5, "common", True),
    ("Firecracker", 3, "common", True),
    ("Royal Giant", 6, "common", True),
    ("Royal Recruits", 7, "common", True),
    ("Cannon", 3, "common", True),
    ("Mortar", 4, "common", True),
    ("Tesla", 4, "common", True),
    ("Zap", 2, "common", True),
    ("Ice Spirit", 1, "common", True),
    ("Bats", 2, "common", True),
    ("Giant Snowball", 2, "common", True),
    ("Minions", 3, "common", False),
    ("Spear Goblins", 2, "common", False),
    ("Goblins", 2, "common", False),
    ("Arrows", 3, "common", False),
    ("Fire Spirit", 1, "common", False),
    ("Skeleton Barrel", 3, "common", False),
    ("Elite Barbarians", 6, "common", False),
    ("Rascals", 5, "common", False),
    # --- rares ---
    ("Musketeer", 4, "rare", False),
    ("Mini P.E.K.K.A", 4, "rare", False),
    ("Valkyrie", 4, "rare", True),
    ("Hog Rider", 4, "rare", False),
    ("Giant", 5, "rare", False),
    ("Wizard", 5, "rare", True),
    ("Battle Ram", 4, "rare", True),
    ("Fireball", 4, "rare", False),
    ("Tombstone", 3, "rare", False),
    ("Inferno Tower", 5, "rare", False),
    ("Flying Machine", 4, "rare", False),
    ("Zappies", 4, "rare", False),
    ("Elixir Collector", 6, "rare", False),
    ("Mega Minion", 3, "rare", False),
    ("Dart Goblin", 3, "rare", False),
    # --- epics ---
    ("Baby Dragon", 4, "epic", False),
    ("P.E.K.K.A", 7, "epic", False),
    ("Witch", 5, "epic", False),
    ("Balloon", 5, "epic", False),
    ("Golem", 8, "epic", False),
    ("Prince", 5, "epic", False),
    ("Dark Prince", 4, "epic", False),
    ("Goblin Barrel", 3, "epic", False),
    ("Skeleton Army", 3, "epic", False),
    ("Lightning", 6, "epic", False),
    ("Poison", 4, "epic", False),
    ("Tornado", 3, "epic", False),
    ("Electro Dragon", 5, "epic", True),
    ("Wall Breakers", 2, "epic", True),
    ("Bowler", 5, "epic", False),
    ("Executioner", 5, "epic", False),
    ("Goblin Drill", 4, "epic", False),
    ("Hunter", 4, "epic", False),
    ("Mother Witch", 4, "epic", False),
    # --- legendaries ---
    ("Mega Knight", 7, "legendary", False),
    ("Electro Wizard", 4, "legendary", False),
    ("Princess", 3, "legendary", False),
    ("Ice Wizard", 3, "legendary", False),
    ("Miner", 3, "legendary", True),
    ("Lava Hound", 7, "legendary", False),
    ("Sparky", 6, "legendary", False),
    ("Inferno Dragon", 4, "legendary", False),
    ("Bandit", 3, "legendary", False),
    ("Royal Ghost", 3, "legendary", False),
    ("Night Witch", 4, "legendary", False),
    ("Magic Archer", 4, "legendary", False),
    ("Ram Rider", 5, "legendary", False),
    ("Fisherman", 3, "legendary", False),
    ("Phoenix", 4, "legendary", False),
    # --- champions ---
    ("Archer Queen", 5, "champion", False),
    ("Golden Knight", 4, "champion", False),
    ("Skeleton King", 4, "champion", False),
    ("Mighty Miner", 4, "champion", False),
    ("Monk", 5, "champion", False),
    ("Little Prince", 3, "champion", False),
]


def build_sample_cards() -> list[Card]:
    cards = []
    for i, (name, elixir, rarity, has_evo) in enumerate(_SAMPLE):
        cards.append(
            build_card(
                id=26_000_000 + i,  # synthetic ids (offline only)
                name=name,
                elixir=elixir,
                rarity=rarity,
                has_evolution=has_evo,
            )
        )
    return cards


if __name__ == "__main__":
    cards = build_sample_cards()
    save_cards_csv(cards)
    print(f"Wrote {len(cards)} sample cards to cards.csv (offline development data).")
