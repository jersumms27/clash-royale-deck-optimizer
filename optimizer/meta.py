"""The meta: the set of opponent decks the GA optimises against (stdlib only).

data/meta_decks.csv holds them with a usage weight each (see data/README.md).
`build_meta_decks` derives the file from battles.csv (most-played decks); the
file can then be edited by hand to pin archetypes or re-weight them.
"""

from __future__ import annotations

import csv
import difflib
from collections import Counter
from dataclasses import dataclass

from optimizer import config
from optimizer.battles import BattleRow, format_id_list, parse_id_list
from optimizer.models import CardPool, Deck


@dataclass(frozen=True)
class MetaDeck:
    cards: tuple[int, ...]
    evo: frozenset[int]
    hero: frozenset[int]
    weight: float
    label: str = ""

    @property
    def key(self) -> tuple:
        return tuple(sorted(self.cards)), tuple(sorted(self.evo)), tuple(sorted(self.hero))


CSV_FIELDS = ["cards", "evo", "hero", "weight", "label"]


def build_meta_decks(
    rows: list[BattleRow],
    n: int = config.META_DECK_COUNT,
    pool: CardPool | None = None,
) -> list[MetaDeck]:
    """Top-`n` most played (cards, evo, hero) combos over both sides of every
    battle, weighted by their share of appearances among the chosen n."""
    counts: Counter[tuple] = Counter()
    for r in rows:
        counts[_key(r.a_cards, r.a_evo, r.a_hero)] += 1
        counts[_key(r.b_cards, r.b_evo, r.b_hero)] += 1
    top = counts.most_common(n)
    total = sum(c for _, c in top) or 1
    return [
        MetaDeck(
            cards=cards,
            evo=frozenset(evo),
            hero=frozenset(hero),
            weight=c / total,
            label=_label(cards, pool),
        )
        for (cards, evo, hero), c in top
    ]


def _key(cards, evo, hero) -> tuple:
    return tuple(sorted(cards)), tuple(sorted(evo)), tuple(sorted(hero))


def _label(cards, pool: CardPool | None) -> str:
    """Two most expensive cards, e.g. 'Golem / Baby Dragon' (needs the pool)."""
    if pool is None:
        return ""
    named = sorted((pool.get(c) for c in cards if c in pool.by_id),
                   key=lambda c: (-c.elixir, c.name))
    return " / ".join(c.name for c in named[:2])


def save_meta_decks(decks: list[MetaDeck], path=config.META_DECKS_CSV) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for d in decks:
            writer.writerow(
                {
                    "cards": format_id_list(d.cards),
                    "evo": format_id_list(d.evo),
                    "hero": format_id_list(d.hero),
                    "weight": f"{d.weight:.6f}",
                    "label": d.label,
                }
            )


def load_meta_decks(
    path=config.META_DECKS_CSV, known_ids: set[int] | None = None
) -> list[MetaDeck]:
    """Read the file; decks with an id outside `known_ids` are dropped and the
    remaining weights re-normalised to sum to 1."""
    decks: list[MetaDeck] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            cards = tuple(parse_id_list(raw.get("cards")))
            if len(cards) != config.DECK_SIZE:
                continue
            if known_ids is not None and not set(cards) <= known_ids:
                continue
            try:
                weight = float(raw.get("weight") or 1.0)
            except ValueError:
                weight = 1.0
            decks.append(
                MetaDeck(
                    cards=cards,
                    evo=frozenset(parse_id_list(raw.get("evo"))) & set(cards),
                    hero=frozenset(parse_id_list(raw.get("hero"))) & set(cards),
                    weight=max(0.0, weight),
                    label=str(raw.get("label") or ""),
                )
            )
    total = sum(d.weight for d in decks)
    if total > 0:
        decks = [MetaDeck(d.cards, d.evo, d.hero, d.weight / total, d.label) for d in decks]
    return decks


def deck_from_names(text: str, pool: CardPool, label: str = "", weight: float = 1.0) -> MetaDeck:
    """Type a deck as card names: "Golem*, Baby Dragon, Knight^, ..." where a
    trailing * marks the evolved form and ^ the hero form (champions need no
    mark). Names are matched case-insensitively; API ids are accepted too."""
    by_name = {c.name.lower(): c for c in pool.cards}
    cards, evo, hero = [], set(), set()
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        is_evo, is_hero = token.endswith("*"), token.endswith("^")
        name = token.rstrip("*^").strip()
        card = by_name.get(name.lower()) or (pool.by_id.get(int(name)) if name.isdigit() else None)
        if card is None:
            close = difflib.get_close_matches(name, [c.name for c in pool.cards], n=3, cutoff=0.5)
            hint = f" (did you mean {', '.join(close)}?)" if close else ""
            raise ValueError(f"unknown card {name!r}{hint}")
        cards.append(card.id)
        if is_evo:
            evo.add(card.id)
        if is_hero and not card.is_champion:
            hero.add(card.id)
    if len(cards) != config.DECK_SIZE or len(set(cards)) != len(cards):
        raise ValueError(f"a deck needs {config.DECK_SIZE} distinct cards, got {len(cards)}")
    return MetaDeck(tuple(cards), frozenset(evo), frozenset(hero), weight, label or _label(cards, pool))


def meta_to_deck(meta: MetaDeck, pool: CardPool) -> Deck:
    """Build a models.Deck for display / scoring. Not validated: real ladder
    decks needn't satisfy the engine's slot repair, and the model doesn't care."""
    champions = {c for c in meta.cards if pool.get(c).is_champion}
    return Deck(
        cards=tuple(pool.get(c) for c in meta.cards),
        evolved=frozenset(meta.evo),
        hero=frozenset(meta.hero) | champions,
    )
