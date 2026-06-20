"""Domain models: Card, CardPool, and Deck (with validity rules)."""

from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass(frozen=True)
class Card:
    # --- core (from the official API) ---
    id: int
    name: str
    elixir: int
    rarity: str
    has_evolution: bool
    # --- scraped combat stats (from cards.csv after scrape.ipynb; None if unknown) ---
    hitpoints: float | None = None
    damage: float | None = None
    damage_per_second: float | None = None
    attack_period: float | None = None
    range: float | None = None
    radius: float | None = None
    lifetime: float | None = None
    crown_tower_damage: float | None = None
    special_damage: float | None = None
    # --- scraped spawn info ---
    troop_spawned: str | None = None
    spawn_count_period: str | None = None
    max_troops_spawned: str | None = None
    # --- scraped evolution info ---
    evo_cycles: float | None = None
    evo_overall_cost: float | None = None
    evo_stat_boosts: str | None = None

    @property
    def is_champion(self) -> bool:
        return self.rarity.lower() == config.CHAMPION_RARITY

    @property
    def is_champion_hero(self) -> bool:
        return self.name in config.CHAMPION_HERO_NAMES

    @property
    def type(self) -> str:
        """'troop', 'building', or 'spell' — from the API id range (with overrides)."""
        if self.name in config.CARD_TYPE_OVERRIDES:
            return config.CARD_TYPE_OVERRIDES[self.name]
        return config.TYPE_BY_ID_PREFIX.get(self.id // 1_000_000, "unknown")

    @property
    def win_condition(self) -> str:
        """'primary', 'secondary', 'conditional', or '' (not a win condition)."""
        return config.WIN_CONDITION_BY_NAME.get(self.name, "")

    @property
    def spell_size(self) -> str:
        """'small' (<=2), 'medium' (3-4), 'large' (5+) for spells; '' otherwise."""
        if self.type != "spell":
            return ""
        if self.elixir <= 2:
            return "small"
        if self.elixir <= 4:
            return "medium"
        return "large"

    @property
    def air(self) -> bool:
        """True if this is an air (flying) troop."""
        return self.name in config.AIR_UNIT_NAMES


class CardPool:
    """The full set of cards to draw from, with lookups the GA needs."""

    def __init__(self, cards: list[Card]):
        self.cards: list[Card] = list(cards)
        self.by_id: dict[int, Card] = {c.id: c for c in self.cards}
        self.all_ids: list[int] = [c.id for c in self.cards]
        self.champion_ids: list[int] = [c.id for c in self.cards if c.is_champion]
        self.champion_set: set[int] = set(self.champion_ids)
        self.non_champion_ids: list[int] = [
            c.id for c in self.cards if not c.is_champion
        ]
        self.evolvable_ids: set[int] = {c.id for c in self.cards if c.has_evolution}
        # Cards eligible for a hero slot (includes champions, which are heroes).
        self.hero_set: set[int] = {c.id for c in self.cards if c.is_champion_hero}

    def __len__(self) -> int:
        return len(self.cards)

    def get(self, card_id: int) -> Card:
        return self.by_id[card_id]


@dataclass(frozen=True)
class Deck:
    """8 cards plus the ids occupying the special slots. Frozen so it's hashable.

    A card's *form* depends on which slot it's in:
      - id in `evolved` -> evolved form (card must have an evolution)
      - id in `hero`    -> hero form    (card must be a champion/hero)
      - otherwise        -> base form
    `evolved` and `hero` are disjoint; champions are always in `hero`. Heroes and
    evolutions share one slot budget (the wild slot), enforced by config.slots_ok.
    """

    cards: tuple[Card, ...]
    evolved: frozenset[int] = field(default_factory=frozenset)
    hero: frozenset[int] = field(default_factory=frozenset)

    @property
    def avg_elixir(self) -> float:
        return sum(c.elixir for c in self.cards) / len(self.cards)

    @property
    def champions(self) -> list[Card]:
        return [c for c in self.cards if c.is_champion]

    @property
    def evolved_cards(self) -> list[Card]:
        return [c for c in self.cards if c.id in self.evolved]

    @property
    def hero_cards(self) -> list[Card]:
        """Cards in hero form (includes champions, which are always hero form)."""
        return [c for c in self.cards if c.id in self.hero]

    def form_of(self, card: Card) -> str:
        """'evo', 'hero', or 'base' -- which form this card takes in the deck."""
        if card.id in self.evolved:
            return "evo"
        if card.id in self.hero:
            return "hero"
        return "base"

    # --- composition by type ---
    @property
    def troops(self) -> list[Card]:
        return [c for c in self.cards if c.type == "troop"]

    @property
    def buildings(self) -> list[Card]:
        return [c for c in self.cards if c.type == "building"]

    @property
    def spells(self) -> list[Card]:
        return [c for c in self.cards if c.type == "spell"]

    @property
    def air_troops(self) -> list[Card]:
        return [c for c in self.cards if c.air]

    # --- win conditions ---
    @property
    def win_conditions(self) -> list[Card]:
        """All cards that are a win condition (any tier)."""
        return [c for c in self.cards if c.win_condition]

    @property
    def primary_win_conditions(self) -> list[Card]:
        return [c for c in self.cards if c.win_condition == "primary"]

    @property
    def card_ids(self) -> set[int]:
        return {c.id for c in self.cards}

    @property
    def key(self) -> tuple:
        """Order-independent identity, used for fitness caching."""
        return (
            tuple(sorted(c.id for c in self.cards)),
            tuple(sorted(self.evolved)),
            tuple(sorted(self.hero)),
        )

    def is_valid(self) -> tuple[bool, str]:
        ids = [c.id for c in self.cards]
        if len(ids) != config.DECK_SIZE:
            return False, f"has {len(ids)} cards, expected {config.DECK_SIZE}"
        if len(set(ids)) != len(ids):
            return False, "has duplicate cards"
        if not config.slots_ok(len(self.evolved), len(self.hero)):
            return False, (
                f"slot rule violated: {len(self.evolved)} evolutions + "
                f"{len(self.hero)} heroes don't fit"
            )
        if self.evolved & self.hero:
            return False, "a card is in both an evolution and a hero slot"

        by_id = {c.id: c for c in self.cards}
        for cid in self.evolved:
            card = by_id.get(cid)
            if card is None:
                return False, "an evolution slot references a card not in the deck"
            if not card.has_evolution:
                return False, f"{card.name} has no evolution available"
        for cid in self.hero:
            card = by_id.get(cid)
            if card is None:
                return False, "a hero slot references a card not in the deck"
            if not card.is_champion_hero:
                return False, f"{card.name} is not a champion/hero"
        # Champions have no base form, so any champion in the deck must be hero form.
        for card in self.cards:
            if card.is_champion and card.id not in self.hero:
                return False, f"{card.name} (champion) must occupy a hero slot"
        return True, "ok"
